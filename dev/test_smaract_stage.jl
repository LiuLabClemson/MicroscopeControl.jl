# Test script for the SmarAct MCS2 stage (MCS2Stage_mod module).
# Run line by line / block by block in the REPL (e.g. Shift+Enter in VS Code).
#
# Units:
#   low-level API  (move!, getposition!, stage.pos_pm, min_pm/max_pm, home_pm)  -> picometres (Int64)
#   generic API    (move, getposition, real_*/targ_*/range_*), move_um!          -> micrometres (Float64)
#
# NOTE: move_all! (used by move!, move_um!, move, home) waits for completion in an
# @async task, so these calls return BEFORE the motion finishes and the position they
# print/store is stale. Use `waitmove(stage)` below before reading the position.

using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.MCS2Stage_mod
const SA = MicroscopeControl.HardwareImplementations.MCS2Stage_mod   # access to non-exported helpers
const SI = MicroscopeControl.HardwareInterfaces.StageInterface

# Block until no connected channel is actively moving (or timeout).
function waitmove(stage; timeout = 30.0)
    t0 = time()
    while any(stage.connected[i] &&
              (SA._get_i32(stage, stage.channel_ids[i], SA.SA_CTL_PKEY_CHANNEL_STATE) &
               SA.SA_CTL_CH_STATE_BIT_ACTIVELY_MOVING) != 0
              for i in 1:stage.n_channels)
        time() - t0 > timeout && (@warn "still moving after $timeout s"; return false)
        sleep(0.05)
    end
    return true
end

## ---------------------------------------------------------------- connect
stage = MCS2Stage(
    stagelabel  = "SmarAct MCS2",
    n_channels  = 3,
    channel_ids = Int32[0, 1, 2],
)
initialize(stage)                      # generic entry point: initialize! + sync of real_*/targ_*/range_*/servostatus

stage.connectionstatus
stage.n_channels                       # truncated if the controller has fewer channels
stage.connected                        # true = positioner attached on that channel
stage.dimensions                       # 3 -> gui uses gui3d
stage.is_calibrated
stage.is_referenced
stage.servostatus

## ---------------------------------------------------------------- software limits
# min_pm/max_pm are the controller's SOFTWARE limits. 0/0 means "no limit configured",
# which makes range_* = (0, 0) and the GUI rejects every move -> set them here (µm).
stage.min_pm, stage.max_pm
stage.range_x, stage.range_y, stage.range_z
stage.range_x = (-20_000.0, 20_000.0)
stage.range_y = (-20_000.0, 20_000.0)
stage.range_z = (-20_000.0, 20_000.0)

## ---------------------------------------------------------------- referencing (only if not referenced)
for i in 1:stage.n_channels
    stage.connected[i] && !stage.is_referenced[i] && find_reference!(stage, i)
end
SA.query_channel_states!(stage)
stage.is_referenced

## ---------------------------------------------------------------- velocity / acceleration (pm/s, pm/s²)
set_velocity!(stage, Int64(1_000_000_000))       # 1 mm/s
set_acceleration!(stage, Int64(1_000_000_000))   # 1 mm/s²
stage.velocity_pm_s, stage.accel_pm_s2

## ---------------------------------------------------------------- read position
getposition!(stage)                    # Vector{Int64}, pm
getposition(stage)                     # updates real_x/real_y/real_z (µm)
stage.real_x, stage.real_y, stage.real_z

## ---------------------------------------------------------------- generic move (µm)
x0, y0, z0 = stage.real_x, stage.real_y, stage.real_z
move(stage, x0 + 10.0, y0, z0); waitmove(stage)
getposition(stage); (stage.real_x, stage.real_y, stage.real_z)

move(stage, x0, y0 + 10.0, z0); waitmove(stage)
getposition(stage); (stage.real_x, stage.real_y, stage.real_z)

move(stage, x0, y0); waitmove(stage)   # 2-argument form (gui2d); z unchanged
getposition(stage); (stage.real_x, stage.real_y, stage.real_z)

## ---------------------------------------------------------------- low-level moves
move_um!(stage, [x0, y0, z0]); waitmove(stage)                 # µm, one value per channel
getposition!(stage) ./ 1e6

p = getposition!(stage)
move!(stage, p .+ Int64(5_000_000)); waitmove(stage)          # +5 µm on every channel (pm)
getposition!(stage) ./ 1e6

SA.move_abs!(stage, 1, p[1])                                  # single channel, blocking (pm)
getposition!(stage) ./ 1e6

## ---------------------------------------------------------------- step scan: commanded vs. measured (channel 1)
targets = x0 .+ collect(-50.0:10.0:50.0)                      # µm
measured = Float64[]
for t in targets
    move(stage, t, y0, z0); waitmove(stage)
    getposition(stage)
    push!(measured, stage.real_x)
end
for (t, m) in zip(targets, measured)
    println("target = ", round(t, digits = 3), "   measured = ", round(m, digits = 3), "   error = ", round(m - t, digits = 4))
end
println("max |error| = ", maximum(abs.(measured .- targets)), " µm")
move(stage, x0, y0, z0); waitmove(stage)

## ---------------------------------------------------------------- stop motion
set_velocity!(stage, Int64(100_000_000))                      # slow down to 0.1 mm/s so there is time to stop
move(stage, x0 + 500.0, y0, z0)                               # returns immediately
sleep(0.5)
stopmotion(stage)                                             # generic; same as stopmotion!(stage) / stop_all!(stage)
getposition(stage); stage.real_x                              # should be between x0 and x0 + 500
stop_channel!(stage, 1)
set_velocity!(stage, Int64(1_000_000_000))
move(stage, x0, y0, z0); waitmove(stage)

## ---------------------------------------------------------------- home
stage.home_pm                                                 # default all zeros (pm)
# stage.home_pm .= round.(Int64, [x0, y0, z0] .* 1e6)         # set home to the start position
home(stage); waitmove(stage)
getposition(stage); (stage.real_x, stage.real_y, stage.real_z)

## ---------------------------------------------------------------- servo (closed loop)
servo(stage, false, false, false)                             # open loop
stage.servostatus
servo(stage, true, true, true)                                # closed loop
stage.servostatus

## ---------------------------------------------------------------- physical travel range (CAUTION: drives into end stops)
# find_travel_range!(stage, 1)                                # needs a referenced channel
# stage.min_pm, stage.max_pm

## ---------------------------------------------------------------- state export
attributes, data, children = export_state(stage)
attributes

## ---------------------------------------------------------------- GUI (gui3d for 3 channels)
SI.gui(stage)

## ---------------------------------------------------------------- disconnect
shutdown(stage)
stage.connectionstatus
