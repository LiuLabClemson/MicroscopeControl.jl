#=
Type definition for the MCL single-axis Nano-Z stage.
=#

"""
    MCLNanoZStage(; kwargs...)

Mad City Labs single-axis Nano-Z piezo stage.

# Keyword arguments
- `stagelabel`: display name (default `"MCL-NanoZ"`)
- `serial`: controller serial number. If `0` (default), the first available
  Nano-Drive is grabbed; set it when more than one MCL controller is attached
  (e.g. an XYZ stage and this Z stage) to pick the right one.
- `axis`: Madlib axis index (1=X, 2=Y, 3=Z, 4=AUX). Auto-detected from the
  product info on `initialize`; the default 3 is used if detection fails.
- `homepos`: position (microns) used by `home`.

Position fields (microns): `real_z` = last read position, `targ_z` = last
commanded position, `range_z` = (0, calibrated travel range).
"""
Base.@kwdef mutable struct MCLNanoZStage <: Stage
    stagelabel::String = "MCL-NanoZ"
    units::String = "Microns"
    id::Int = 0                                     # device handle
    serial::Int = 0                                 # 0 = grab first available device
    axis::Int = 3                                   # Madlib axis index, auto-detected
    dimensions::Int = 1
    connectionstatus::Bool = false
    real_z::Float64 = 0.0
    targ_z::Float64 = 0.0
    range_z::Tuple{Float64,Float64} = (0.0, 100.0)
    homepos::Float64 = 0.0
end
