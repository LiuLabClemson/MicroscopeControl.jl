module SmarAct_MCS2

using ...MicroscopeControl.HardwareInterfaces.StageInterface
import ...MicroscopeControl: export_state, initialize, shutdown

# Path to the SmarAct DLL
const SmarAct = "C:\\Windows\\System32\\SmarActCTL.dll"

include("constants_smaract.jl")   # SA_CTL_* constants and type aliases
include("functions_smaract.jl")   # ccall wrappers for every SDK function
include("types.jl")               # MCS2Stage struct + constructor
include("helper.jl")              # pm-level device / motion helpers
include("interface_methods.jl")   # StageInterface methods (µm)

export MCS2Stage

# StageInterface methods (µm)
export move, getposition, getrange, home, stopmotion, servo, gui

# Low-level helpers (pm)
export getposition_pm, move_all!
export find_reference!, find_travel_range!
export set_velocity!, set_acceleration!
export query_positions!, stop_all!, stop_channel!

end
