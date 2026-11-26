#=
Stage implementation for the Mad City Labs single-axis Nano-Z piezo stage.

Uses the same Madlib.dll as the MadCityLabs (Nano-XYZ) module. The low-level
ccall wrappers live in functions_madlib.jl and the C constants/structs in
constants_madlib.jl (same layout as pi_n472).
=#

module MCLNanoZ
    using ...MicroscopeControl.HardwareInterfaces.StageInterface

    import ...MicroscopeControl: export_state, initialize, shutdown

    const MadLib = "C:\\Program Files\\Mad City Labs\\NanoDrive\\API\\Madlib.dll"

    include("constants_madlib.jl")
    include("functions_madlib.jl")
    include("types.jl")
    include("helper.jl")
    include("interface_methods.jl")   # gui(stage) uses StageInterface.gui1d with the *_z fields

    export MCLNanoZStage
    export move, getposition, getrange, home, stopmotion, gui
    export nanoz_isattached, nanoz_printinfo, nanoz_monitor
end
