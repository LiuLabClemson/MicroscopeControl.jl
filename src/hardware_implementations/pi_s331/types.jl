#=
Type definition for the PI S-331.2SH tip/tilt mirror (E-727.3SDA controller).
=#

"""
    S331(; kwargs...)

PI S-331.2SH fast tip/tilt piezo mirror controlled by an E-727.3SDA over USB.

The mirror axes are stored as a 1D or 2D stage: `x` = first axis,
`y` = second axis (unused when `dimensions == 1`). Positions are tilt angles
in the controller's units (queried on `initialize`, typically µrad).

# Keyword arguments
- `stagelabel`: display name (default `"PI-S331"`)
- `controllername`: USB enumeration filter (default `"E-727"`)
- `serial`: if non-empty, pick the USB device whose description contains this
  string (use when several E-727 controllers are attached)
- `axes`: GCS axis identifiers to control, e.g. `["A", "B"]` or `["1"]`.
  If empty (default), the first two axes reported by the controller (`SAI?`)
  are used (or one, if the controller only reports one).
- `homepos`: tilt used by `home` (one value per axis). If empty (default),
  the center of the travel range is used.
- `settle_timeout`: max time (s) `move` waits for the on-target flag
- `voltage_channels`: piezo output channels read by `getvoltage`
"""
Base.@kwdef mutable struct S331 <: Stage
    stagelabel::String = "PI-S331"
    controllername::String = "E-727"
    serial::String = ""
    idn::String = ""
    units::String = "urad"
    id::Int = -1                                       # GCS connection ID (-1 = not connected)
    dimensions::Int = 2                                # set from length(axes) on initialize
    axes::Vector{String} = String[]                    # auto-detected if empty
    connectionstatus::Bool = false
    real_x::Float64 = 0.0
    real_y::Float64 = 0.0
    targ_x::Float64 = 0.0
    targ_y::Float64 = 0.0
    range_x::Tuple{Float64,Float64} = (0.0, 0.0)
    range_y::Tuple{Float64,Float64} = (0.0, 0.0)
    servostatus::Tuple{Bool,Bool} = (false, false)
    homepos::Vector{Float64} = Float64[]               # center of range if empty
    settle_timeout::Float64 = 1.0
    voltage_channels::Vector{Int} = [1, 2, 3]
end
