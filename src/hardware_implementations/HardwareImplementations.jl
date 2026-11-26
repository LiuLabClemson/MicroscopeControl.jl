"""
HardwareImplementations module is a container for all hardware implementations.
Uses Reexport.jl to automatically propagate exports from submodules.
"""
module HardwareImplementations

using Reexport
using ..MicroscopeControl

# Camera implementations
include("simulated_camera/SimulatedCamera.jl")
@reexport using .SimulatedCamera

include("dcam4_camera/DCAM4.jl")
@reexport using .DCAM4

include("thorcam_csc/ThorCamCSC.jl")
@reexport using .ThorCamCSC

include("thorcam_dcx/ThorCamDCx.jl")
@reexport using .ThorCamDCx

include("ids_camera/IDSCam.jl")
@reexport using .IDSCam

# Stage implementations
include("pi_stage/PI.jl")
@reexport using .PI

include("simulated_stage/SimulatedStage.jl")
@reexport using .SimulatedStage

include("mcl_stage/MadCityLabs.jl")
@reexport using .MadCityLabs

include("mcl_nano_z/MCLNanoZ.jl")
@reexport using .MCLNanoZ

include("pi_N472/PI_N472.jl")
@reexport using .PI_N472

include("pi_s331/PI_S331.jl")   # reuses the GCS2 bindings from PI_N472 (must come after it)
@reexport using .PI_S331

include("smaract_stage/MCS2Stage_module.jl")
@reexport using .MCS2Stage_mod

# DAQ implementation (must come before modules that depend on it)
include("nidaq/NIDAQcard.jl")
@reexport using .NIDAQcard

# Light source implementations
include("simulated_light/SimulatedLight.jl")
@reexport using .SimulatedLight

include("tcube_laser/TCubeLaserControl.jl")
@reexport using .TCubeLaserControl

include("daq_transmission_light/TransmissionDaqControl.jl")
@reexport using .TransmissionDaqControl

include("crysta_laser_561/CrystaLaserControl.jl")
@reexport using .CrystaLaserControl

include("vortran_laser_488/VortranLaserControl.jl")
@reexport using .VortranLaserControl

# FPGA implementation (depends on NIDAQcard)
include("ok_xem/OK_XEM.jl")
@reexport using .OK_XEM

# SLM implementation
include("meadowlark_slm/Meadowlark.jl")
@reexport using .Meadowlark

# FPGA DAC implementation
include("xem_dac/XEM_DAC.jl")
@reexport using .XEM_DAC

# Work in progress - uncomment when ready
# include("triggerscope/Triggerscope.jl")
# @reexport using .Triggerscope

# include("mcl_micro_positioner/MCLMicroPositioner.jl")
# @reexport using .MCLMicroPositioner

end
