#=
StageInterface methods for the SmarAct MCS2. Positions are in micrometres
(real_x/y/z, targ_x/y/z, range_x/y/z); the pm-level work is in helper.jl.
=#

"""
    initialize(stage::MCS2Stage)

Open the MCS2, configure all connected channels, and populate the µm fields
(`real_*`, `targ_*`, `range_*`) and `servostatus` from the hardware.
"""
function initialize(stage::MCS2Stage)
    if stage.connectionstatus
        @error "Stage already initialized — call shutdown(stage) first."
        return
    end
    open_device!(stage) || return

    stage.range_x = (stage.min_pm[1] / PM_PER_UM, stage.max_pm[1] / PM_PER_UM)
    if stage.n_channels >= 2
        stage.range_y = (stage.min_pm[2] / PM_PER_UM, stage.max_pm[2] / PM_PER_UM)
    end
    if stage.n_channels >= 3
        stage.range_z = (stage.min_pm[3] / PM_PER_UM, stage.max_pm[3] / PM_PER_UM)
    end

    _sync_real_um!(stage)
    stage.targ_x, stage.targ_y, stage.targ_z = stage.real_x, stage.real_y, stage.real_z

    # Sync servostatus (closed-loop on/off) from the hardware for every connected channel
    for i in 1:stage.n_channels
        stage.connected[i] || continue
        stage.servostatus[i] = _get_i32(stage, stage.channel_ids[i], SA_CTL_PKEY_CONTROL_LOOP_INPUT) != SA_CTL_CONTROL_LOOP_INPUT_DISABLED
    end

    @info "$(stage.stagelabel) initialized ($(stage.n_channels) channel(s))."
end

"""
    shutdown(stage::MCS2Stage)

Close the device connection. Safe to call even if not connected.
"""
function shutdown(stage::MCS2Stage)
    if !stage.connectionstatus
        @info "Stage not connected — nothing to shut down."
        return
    end
    result = SA_CTL_Close(stage.dHandle[])
    if result != SA_CTL_ERROR_NONE
        @warn "SA_CTL_Close error: $(unsafe_string(SA_CTL_GetResultInfo(result)))"
    else
        @info "$(stage.stagelabel) disconnected."
    end
    stage.connectionstatus = false
end

"""
    move(stage::MCS2Stage, x, y)
    move(stage::MCS2Stage, x, y, z)
    move(stage::MCS2Stage, targets_um::AbstractVector)

Absolute move in micrometres (one value per channel for the vector form).
Channels with no positioner attached are ignored. Returns before the motion
finishes so it can be interrupted with `stopmotion`; call `getposition`
once it has settled. For picometre targets use `move_all!(stage, targets_pm)`.
"""
StageInterface.move(stage::MCS2Stage, x::Real, y::Real) = _apply_move!(stage, Float64[x, y])
StageInterface.move(stage::MCS2Stage, x::Real, y::Real, z::Real) = _apply_move!(stage, Float64[x, y, z])
StageInterface.move(stage::MCS2Stage, targets_um::AbstractVector{<:Real}) = _apply_move!(stage, targets_um)

"""
    getposition(stage::MCS2Stage)

Read all channels, update `pos_pm` and `real_x`/`real_y`/`real_z`, and return
the position in µm as `(x, y)` or `(x, y, z)`. Use `getposition_pm(stage)` for picometres.
"""
function StageInterface.getposition(stage::MCS2Stage)
    query_positions!(stage)
    _sync_real_um!(stage)
    return stage.n_channels >= 3 ? (stage.real_x, stage.real_y, stage.real_z) : (stage.real_x, stage.real_y)
end

"""
    getrange(stage::MCS2Stage)

Software travel range in µm, `(range_x, range_y)` or `(range_x, range_y, range_z)`.
"""
function StageInterface.getrange(stage::MCS2Stage)
    return stage.n_channels >= 3 ? (stage.range_x, stage.range_y, stage.range_z) : (stage.range_x, stage.range_y)
end

"""
    home(stage::MCS2Stage)

Move all channels to `stage.home_pm` (default: all zeros).
"""
function StageInterface.home(stage::MCS2Stage)
    @info "Moving to home position ..."
    return _apply_move!(stage, stage.home_pm ./ PM_PER_UM)
end

"""
    stopmotion(stage::MCS2Stage)

Immediately stop all connected channels.
"""
StageInterface.stopmotion(stage::MCS2Stage) = stop_all!(stage)

"""
    servo(stage::MCS2Stage, xtoggle, ytoggle)
    servo(stage::MCS2Stage, xtoggle, ytoggle, ztoggle)

Switch closed-loop control on/off per channel. Channels with no positioner are skipped.
"""
StageInterface.servo(stage::MCS2Stage, xtoggle::Bool, ytoggle::Bool) = _apply_servo!(stage, Bool[xtoggle, ytoggle])
StageInterface.servo(stage::MCS2Stage, xtoggle::Bool, ytoggle::Bool, ztoggle::Bool) = _apply_servo!(stage, Bool[xtoggle, ytoggle, ztoggle])

"""
    export_state(stage::MCS2Stage) -> (attributes, data, children)
"""
function export_state(stage::MCS2Stage)
    attributes = Dict{String, Any}(
        "stage_label"     => stage.stagelabel,
        "n_channels"      => stage.n_channels,
        "channel_ids"     => copy(stage.channel_ids),
        "connected"       => stage.connectionstatus,
        "channel_connected" => copy(stage.connected),
        "handle"          => stage.dHandle[],
        "position_pm"     => copy(stage.pos_pm),
        "position_um"     => stage.pos_pm ./ PM_PER_UM,
        "min_pm"          => copy(stage.min_pm),
        "max_pm"          => copy(stage.max_pm),
        "home_pm"         => copy(stage.home_pm),
        "velocity_pm_s"   => copy(stage.velocity_pm_s),
        "accel_pm_s2"     => copy(stage.accel_pm_s2),
        "is_calibrated"   => copy(stage.is_calibrated),
        "is_referenced"   => copy(stage.is_referenced),
        "servo_status"    => copy(stage.servostatus),
    )
    return attributes, nothing, Dict{String,Any}()
end
