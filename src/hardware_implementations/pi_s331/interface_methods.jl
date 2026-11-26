"""
    initialize(mirror::S331)

Connect to the E-727 over USB, detect the axes, query units and travel range,
and switch the servo (closed loop) on. Returns the GCS connection ID (< 0 on failure).
"""
function initialize(mirror::S331)
    if mirror.connectionstatus
        @warn "$(mirror.stagelabel) already initialized"
        return mirror.id
    end

    devices = enumerate_devices(mirror.controllername)
    if isempty(devices)
        @error "No $(mirror.controllername) controller found on USB"
        return -1
    end
    if !isempty(mirror.serial)
        devices = filter(d -> occursin(mirror.serial, d), devices)
        if isempty(devices)
            @error "No $(mirror.controllername) with serial '$(mirror.serial)' found"
            return -1
        end
    elseif length(devices) > 1
        @warn "Several controllers found, using the first one (set serial= to choose)" devices
    end

    id = PI_ConnectUSB(devices[1])
    if id < 0
        @error "Failed to connect to $(devices[1])" code = PI_GetInitError()
        return id
    end
    mirror.id = id
    mirror.connectionstatus = true
    queryidn!(mirror)

    # ---- axis detection
    available = queryaxes(mirror)
    if isempty(mirror.axes)
        if isempty(available)
            @error "Controller reported no axes"
            shutdown(mirror)
            return -1
        end
        mirror.axes = available[1:min(2, end)]
        if length(available) > 2
            @info "Controller reports axes $available; using $(mirror.axes). Pass axes=[...] to choose others."
        end
    else
        missing_axes = setdiff(mirror.axes, available)
        if !isempty(missing_axes) || !(1 <= length(mirror.axes) <= 2)
            @error "Invalid axes $(mirror.axes); controller reports $available (use 1 or 2 of them)"
            shutdown(mirror)
            return -1
        end
    end
    mirror.dimensions = naxes(mirror)

    queryunits!(mirror)
    queryrange!(mirror)
    setservo!(mirror, true, true)
    readposition!(mirror)
    readtarget!(mirror)

    @info "$(mirror.stagelabel) initialized" device = devices[1] idn = mirror.idn axes = mirror.axes units = mirror.units range_x = mirror.range_x range_y = mirror.range_y servo = mirror.servostatus
    return id
end

"""
    shutdown(mirror::S331)

Close the USB connection. The servo is left as is so the mirror does not jump.
"""
function shutdown(mirror::S331)
    if mirror.id >= 0 && PI_IsConnected(mirror.id) == TRUE
        PI_CloseConnection(mirror.id)
        @info "$(mirror.stagelabel) disconnected"
    else
        @info "$(mirror.stagelabel) not connected"
    end
    mirror.connectionstatus = false
    mirror.id = -1
    return nothing
end

"""
    move(mirror::S331, x, y; wait = true)
    move(mirror::S331, x; wait = true)
    move(mirror::S331, pos::AbstractVector; wait = true)

Move to an absolute tilt (controller units, see `mirror.units`). With one
argument on a 2-axis mirror only the first axis changes. If `wait`, block
until on target (or `mirror.settle_timeout`). Returns `true` on success.
"""
function StageInterface.move(mirror::S331, x::Real, y::Real; wait::Bool=true)
    return move_abs!(mirror, x, y; wait=wait)
end

function StageInterface.move(mirror::S331, x::Real; wait::Bool=true)
    return move_abs!(mirror, x, mirror.targ_y; wait=wait)
end

function StageInterface.move(mirror::S331, pos::AbstractVector{<:Real}; wait::Bool=true)
    length(pos) == naxes(mirror) || throw(ArgumentError("expected $(naxes(mirror)) values, got $(length(pos))"))
    return move_abs!(mirror, pos[1], naxes(mirror) == 2 ? pos[2] : mirror.targ_y; wait=wait)
end

"""
    moverelative(mirror::S331, dx, dy = 0.0; wait = true)

Move relative to the last commanded target.
"""
function moverelative(mirror::S331, dx::Real, dy::Real=0.0; wait::Bool=true)
    return move_abs!(mirror, mirror.targ_x + dx, mirror.targ_y + dy; wait=wait)
end

"""
    getposition(mirror::S331)

Read the sensor position. Returns `x` (1 axis) or `(x, y)` (2 axes); also
stored in `mirror.real_x` / `mirror.real_y`.
"""
function StageInterface.getposition(mirror::S331)
    readposition!(mirror)
    return naxes(mirror) == 1 ? mirror.real_x : (mirror.real_x, mirror.real_y)
end

"""
    getrange(mirror::S331)

Return the travel range `range_x` (1 axis) or `(range_x, range_y)` (2 axes).
"""
function StageInterface.getrange(mirror::S331)
    queryrange!(mirror)
    return naxes(mirror) == 1 ? mirror.range_x : (mirror.range_x, mirror.range_y)
end

"""
    home(mirror::S331)

Move to `mirror.homepos`, or to the center of the travel range if it is empty.
"""
function StageInterface.home(mirror::S331)
    if isempty(mirror.homepos)
        x = sum(mirror.range_x) / 2
        y = sum(mirror.range_y) / 2
    else
        x = mirror.homepos[1]
        y = length(mirror.homepos) > 1 ? mirror.homepos[2] : mirror.targ_y
    end
    @info "$(mirror.stagelabel) homing to $(naxes(mirror) == 1 ? x : (x, y)) $(mirror.units)"
    return move_abs!(mirror, x, y)
end

function StageInterface.stopmotion(mirror::S331)
    ok = checkerror(mirror, PI_HLT(mirror.id, axisstring(mirror)), "HLT")
    clearerror!(mirror)          # HLT sets error 10 ("stopped by command")
    readtarget!(mirror)
    return ok
end

"""
    servo(mirror::S331, xon::Bool, yon::Bool)
    servo(mirror::S331, on::Bool)

Switch closed-loop control on/off. Returns the new servo state.
"""
StageInterface.servo(mirror::S331, xon::Bool, yon::Bool) = setservo!(mirror, xon, yon)
StageInterface.servo(mirror::S331, on::Bool) = setservo!(mirror, on, on)

function StageInterface.driftcorrection(mirror::S331, args...)
    @warn "Drift correction not supported by $(mirror.stagelabel)"
    return nothing
end

"""
    gui(mirror::S331)

Uses the generic 2D stage GUI (x = first axis, y = second axis).
"""
function StageInterface.gui(mirror::S331)
    if mirror.dimensions == 2
        return StageInterface.gui2d(mirror)
    end
    @error "GUI only supports the 2-axis configuration; use move/getposition for $(mirror.dimensions) axis"
    return nothing
end

"""
    autozero(mirror::S331; timeout = 60) -> Bool

Run the AutoZero procedure (`ATZ`) on the selected axes with the default low
voltage. The mirror moves through its full range while this runs and the
servo is switched off for it. Only needed after the setup changed; the result
is stored on the controller.
"""
function autozero(mirror::S331; timeout::Real=60.0)
    n = naxes(mirror)
    servo_before = mirror.servostatus
    @warn "$(mirror.stagelabel): running AutoZero, the mirror will move through its full range"
    setservo!(mirror, false, false)

    ok = checkerror(mirror, PI_ATZ(mirror.id, axisstring(mirror), zeros(Float64, n), fill(TRUE, n)), "ATZ") &&
         waitready(mirror; timeout=timeout)

    result = zeros(Cint, n)
    if ok && checkerror(mirror, PI_qATZ(mirror.id, axisstring(mirror), result), "qATZ")
        ok = all(==(1), result)
        ok || @error "AutoZero failed" axes = mirror.axes result
    end

    setservo!(mirror, servo_before...)
    readposition!(mirror)
    readtarget!(mirror)
    return ok
end

"""
    getvoltage(mirror::S331; channels = mirror.voltage_channels)

Read the piezo output voltages (V) of the given controller output channels.
"""
function getvoltage(mirror::S331; channels::AbstractVector{<:Integer}=mirror.voltage_channels)
    ch = Cint.(channels)
    v = zeros(Float64, length(ch))
    checkerror(mirror, PI_qVOL(mirror.id, ch, v, Cint(length(ch))), "qVOL") || return nothing
    return v
end

"""
    s331_printinfo(mirror::S331)

Print what the controller reports (USB devices, IDN, axes, units, ranges,
servo, position, voltages). Useful to find out the axis configuration.
"""
function s331_printinfo(mirror::S331)
    println("USB devices ($(mirror.controllername)): ", enumerate_devices(mirror.controllername))
    if !mirror.connectionstatus
        println("Not connected; call initialize(mirror) for more information.")
        return nothing
    end
    println("IDN:              ", queryidn!(mirror))
    println("Axes (SAI?):      ", queryaxes(mirror))
    println("Axes (SAI? ALL):  ", queryaxes(mirror; all=true))
    println("Used axes:        ", mirror.axes, "  (dimensions = $(mirror.dimensions))")
    println("Units:            ", queryunits!(mirror))
    println("Range:            ", getrange(mirror))
    println("Servo:            ", queryservo!(mirror))
    println("Position:         ", getposition(mirror))
    println("Target:           ", readtarget!(mirror)[1:naxes(mirror)])
    println("Voltages (ch $(mirror.voltage_channels)): ", getvoltage(mirror))
    return nothing
end

"""
    export_state(mirror::S331)
"""
function export_state(mirror::S331)
    attributes = Dict{String,Any}(
        "stage_label" => mirror.stagelabel,
        "controller" => mirror.controllername,
        "idn" => mirror.idn,
        "units" => mirror.units,
        "id" => mirror.id,
        "dimensions" => mirror.dimensions,
        "axes" => join(mirror.axes, " "),
        "connected" => mirror.connectionstatus,
        "position_x" => mirror.real_x,
        "position_y" => mirror.real_y,
        "target_x" => mirror.targ_x,
        "target_y" => mirror.targ_y,
        "range_x" => collect(mirror.range_x),
        "range_y" => collect(mirror.range_y),
        "servo_status" => collect(mirror.servostatus),
        "home_position" => copy(mirror.homepos),
    )

    data = nothing
    children = Dict{String,Any}()

    return attributes, data, children
end
