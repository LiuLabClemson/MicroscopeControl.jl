# Test script for the PI S-331.2SH tip/tilt mirror on an E-727.3SDA (PI_S331 module).
# Run line by line / block by block in the REPL (e.g. Shift+Enter in VS Code).

using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.PI_S331
const S = MicroscopeControl.HardwareImplementations.PI_S331   # access to non-exported helpers

## ---------------------------------------------------------------- connect
mirror = S331()                        # axes auto-detected; or S331(axes = ["A", "B"]) / S331(serial = "1234")
initialize(mirror)

s331_printinfo(mirror)                 # <-- shows how many axes the controller reports, units, range
mirror.axes                            # axes actually used
mirror.dimensions                      # 1 or 2
mirror.units                           # usually µrad
mirror.range_x, mirror.range_y

## ---------------------------------------------------------------- basic move / read
getposition(mirror)
home(mirror)                           # center of range
getposition(mirror)

cx = sum(mirror.range_x) / 2
cy = sum(mirror.range_y) / 2
move(mirror, cx + 100, cy)             # tilt first axis
getposition(mirror)
move(mirror, cx, cy + 100)             # tilt second axis
getposition(mirror)
moverelative(mirror, -50, -50)
getposition(mirror)

move(mirror, 1.0e9, 0.0)               # should be rejected: out of range

## ---------------------------------------------------------------- step scan: commanded vs. measured (first axis)
targets = collect(range(mirror.range_x[1] + 0.1 * (mirror.range_x[2] - mirror.range_x[1]),
                        mirror.range_x[2] - 0.1 * (mirror.range_x[2] - mirror.range_x[1]), length = 9))
measured = Float64[]
for t in targets
    move(mirror, t, cy)
    push!(measured, mirror.dimensions == 1 ? getposition(mirror) : getposition(mirror)[1])
end
for (t, m) in zip(targets, measured)
    println("target = ", round(t, digits = 3), "   measured = ", round(m, digits = 3), "   error = ", round(m - t, digits = 4))
end

## ---------------------------------------------------------------- settle time of a step
home(mirror)
t0 = time(); move(mirror, cx + 200, cy); println("step + settle: ", round(1e3 * (time() - t0), digits = 2), " ms")

## ---------------------------------------------------------------- servo / voltages
getvoltage(mirror)
servo(mirror, false, false)            # open loop
servo(mirror, true, true)              # closed loop
# autozero(mirror)                     # only if needed: sweeps the mirror through its full range

## ---------------------------------------------------------------- state export
attributes, data, children = export_state(mirror)
attributes

## ---------------------------------------------------------------- GUI (2-axis only)
gui(mirror)

## ---------------------------------------------------------------- disconnect
home(mirror)
shutdown(mirror)
