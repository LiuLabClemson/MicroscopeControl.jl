#=
Low-level helpers for the S-331 mirror built on the GCS2 DLL wrappers.
All per-axis arrays have length `naxes(mirror)` (1 or 2).
=#

const BUFSIZE = 1024

"""Convert a NUL-terminated byte buffer to a String (Latin-1, so `µ` survives)."""
function cstring(buf::Vector{UInt8})
    n = something(findfirst(iszero, buf), length(buf) + 1) - 1
    return String(Char.(buf[1:n]))
end

nonempty(parts) = [String(strip(p)) for p in parts if !isempty(strip(p))]

naxes(mirror::S331) = length(mirror.axes)
axisstring(mirror::S331) = join(mirror.axes, " ")

"""Per-axis values as a vector of length naxes (x first, then y)."""
xy(mirror::S331, x, y) = naxes(mirror) == 1 ? Float64[x] : Float64[x, y]

"""Translate a GCS error code into its message."""
function errormessage(code::Integer)
    buf = zeros(UInt8, BUFSIZE)
    PI_TranslateError(Cint(code), buf, BUFSIZE)
    return cstring(buf)
end

"""
    checkerror(mirror, success, what) -> Bool

If a DLL call returned FALSE, log the controller error and return `false`.
"""
function checkerror(mirror::S331, success, what::AbstractString)
    success == TRUE && return true
    code = PI_GetError(mirror.id)
    @error "$(mirror.stagelabel): $what failed" code message = errormessage(code)
    return false
end

"""List the USB descriptions of attached controllers matching `filter`."""
function enumerate_devices(filter::AbstractString)
    buf = zeros(UInt8, BUFSIZE)
    n = PI_EnumerateUSB(buf, BUFSIZE, filter)
    n <= 0 && return String[]
    return nonempty(split(cstring(buf), '\n'))
end

function queryidn!(mirror::S331)
    buf = zeros(UInt8, BUFSIZE)
    if checkerror(mirror, PI_qIDN(mirror.id, buf, BUFSIZE), "qIDN")
        mirror.idn = strip(cstring(buf))
    end
    return mirror.idn
end

"""Axis identifiers reported by the controller (`SAI?`, or `SAI? ALL` if `all=true`)."""
function queryaxes(mirror::S331; all::Bool=false)
    buf = zeros(UInt8, BUFSIZE)
    ok = all ? PI_qSAI_ALL(mirror.id, buf, BUFSIZE) : PI_qSAI(mirror.id, buf, BUFSIZE)
    checkerror(mirror, ok, all ? "qSAI_ALL" : "qSAI") || return String[]
    return nonempty(split(cstring(buf)))
end

function queryunits!(mirror::S331)
    buf = zeros(UInt8, BUFSIZE)
    if checkerror(mirror, PI_qPUN(mirror.id, axisstring(mirror), buf, BUFSIZE), "qPUN")
        # answer looks like "A=µrad\nB=µrad\n"; keep the unit of the first axis
        lines = nonempty(split(cstring(buf), '\n'))
        isempty(lines) || (mirror.units = String(strip(last(split(lines[1], '=')))))
    end
    return mirror.units
end

function queryrange!(mirror::S331)
    n = naxes(mirror)
    lo = zeros(Float64, n)
    hi = zeros(Float64, n)
    ok = checkerror(mirror, PI_qTMN(mirror.id, axisstring(mirror), lo), "qTMN") &&
         checkerror(mirror, PI_qTMX(mirror.id, axisstring(mirror), hi), "qTMX")
    if ok
        mirror.range_x = (lo[1], hi[1])
        n == 2 && (mirror.range_y = (lo[2], hi[2]))
    end
    return mirror.range_x, mirror.range_y
end

function queryservo!(mirror::S331)
    state = zeros(BOOL, naxes(mirror))
    if checkerror(mirror, PI_qSVO(mirror.id, axisstring(mirror), state), "qSVO")
        on = state .== TRUE
        mirror.servostatus = (on[1], naxes(mirror) == 2 ? on[2] : on[1])
    end
    return mirror.servostatus
end

function setservo!(mirror::S331, xon::Bool, yon::Bool)
    state = BOOL[b ? TRUE : FALSE for b in (xon, yon)][1:naxes(mirror)]
    checkerror(mirror, PI_SVO(mirror.id, axisstring(mirror), state), "SVO")
    return queryservo!(mirror)
end

function readposition!(mirror::S331)
    pos = zeros(Float64, naxes(mirror))
    if checkerror(mirror, PI_qPOS(mirror.id, axisstring(mirror), pos), "qPOS")
        mirror.real_x = pos[1]
        naxes(mirror) == 2 && (mirror.real_y = pos[2])
    end
    return mirror.real_x, mirror.real_y
end

function readtarget!(mirror::S331)
    pos = zeros(Float64, naxes(mirror))
    if checkerror(mirror, PI_qMOV(mirror.id, axisstring(mirror), pos), "qMOV")
        mirror.targ_x = pos[1]
        naxes(mirror) == 2 && (mirror.targ_y = pos[2])
    end
    return mirror.targ_x, mirror.targ_y
end

inrange(v, r) = r[1] <= v <= r[2]

"""
    waitontarget(mirror; timeout = mirror.settle_timeout) -> Bool

Poll `ONT?` until all axes report on-target or `timeout` seconds pass.
"""
function waitontarget(mirror::S331; timeout::Real=mirror.settle_timeout)
    ont = zeros(BOOL, naxes(mirror))
    t0 = time()
    while true
        checkerror(mirror, PI_qONT(mirror.id, axisstring(mirror), ont), "qONT") || return false
        all(==(TRUE), ont) && return true
        if time() - t0 > timeout
            @warn "$(mirror.stagelabel): not on target after $(timeout) s" ontarget = Bool.(ont)
            return false
        end
    end
end

"""
    move_abs!(mirror, x, y; wait = true) -> Bool

Command an absolute tilt (`y` ignored for a 1-axis setup).
Rejects targets outside the travel range.
"""
function move_abs!(mirror::S331, x::Real, y::Real; wait::Bool=true)
    if !mirror.connectionstatus
        @error "$(mirror.stagelabel) not connected"
        return false
    end
    twoaxes = naxes(mirror) == 2
    if !inrange(x, mirror.range_x) || (twoaxes && !inrange(y, mirror.range_y))
        @error "$(mirror.stagelabel): target out of range" x y range_x = mirror.range_x range_y = mirror.range_y
        return false
    end
    if !all(mirror.servostatus)
        @warn "$(mirror.stagelabel): servo is off; MOV needs closed loop (call servo(mirror, true, true))"
    end

    target = xy(mirror, x, y)
    checkerror(mirror, PI_MOV(mirror.id, axisstring(mirror), target), "MOV") || return false
    mirror.targ_x = x
    twoaxes && (mirror.targ_y = y)
    return wait ? waitontarget(mirror) : true
end

"""Wait until the controller is ready (e.g. after ATZ)."""
function waitready(mirror::S331; timeout::Real=30.0)
    ready = Cint[0]
    t0 = time()
    while ready[1] == 0
        checkerror(mirror, PI_IsControllerReady(mirror.id, ready), "IsControllerReady") || return false
        if time() - t0 > timeout
            @error "$(mirror.stagelabel): controller not ready after $(timeout) s"
            return false
        end
        sleep(0.05)
    end
    return true
end

"""Read and clear the controller error register (`ERR?`)."""
function clearerror!(mirror::S331)
    code = Cint[0]
    PI_qERR(mirror.id, code)
    return code[1]
end
