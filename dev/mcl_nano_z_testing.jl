# Test script for the MCL single-axis Nano-Z stage (MCLNanoZ module).
# Run line by line / block by block in the REPL (e.g. Shift+Enter in VS Code).

using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.MCLNanoZ
const NZ = MicroscopeControl.HardwareImplementations.MCLNanoZ   # access to non-exported helpers

## ---------------------------------------------------------------- connect
zstage = MCLNanoZStage()               # or MCLNanoZStage(serial = XXXX) if several MCL controllers are attached
initialize(zstage)

zstage.id                              # handle (> 0 if connected)
zstage.axis                            # detected Madlib axis (1=X, 2=Y, 3=Z)
zstage.range_z                         # calibrated travel range (microns)
nanoz_isattached(zstage)
nanoz_printinfo(zstage)
NZ.getserialnumber(zstage)
NZ.getproductinfo(zstage)

## ---------------------------------------------------------------- basic move / read
getposition(zstage)
move(zstage, 10.0)
sleep(0.1)
getposition(zstage)

nanoz_monitor(zstage, 20.0)            # move and read back in one call

move(zstage, 1.0e6)                    # should be rejected: out of range

home(zstage)
getposition(zstage)

## ---------------------------------------------------------------- step scan: commanded vs. measured
zmax = zstage.range_z[2]
targets = collect(range(0.1 * zmax, 0.9 * zmax, length = 9))
measured = Float64[]
for zt in targets
    move(zstage, zt)
    sleep(0.05)                        # settle time
    push!(measured, getposition(zstage))
end
errors = measured .- targets
for (t, m, e) in zip(targets, measured, errors)
    println("target = ", round(t, digits = 3), "   measured = ", round(m, digits = 3), "   error = ", round(e, digits = 4))
end
println("max |error| = ", maximum(abs.(errors)), " ", zstage.units)

## ---------------------------------------------------------------- small steps (resolution check)
start = zstage.range_z[2] / 2
move(zstage, start); sleep(0.1)
for i in 0:10
    move(zstage, start + 0.01 * i)     # 10 nm steps
    sleep(0.05)
    println("step ", i, ": ", round(getposition(zstage), digits = 4))
end

## ---------------------------------------------------------------- state export
attributes, data, children = export_state(zstage)
attributes

## ---------------------------------------------------------------- GUI
gui(zstage)

## ---------------------------------------------------------------- disconnect
home(zstage)
shutdown(zstage)
zstage.connectionstatus
