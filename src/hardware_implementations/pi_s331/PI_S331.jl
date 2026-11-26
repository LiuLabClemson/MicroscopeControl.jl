#=
Stage implementation for the PI S-331.2SH fast tip/tilt piezo mirror driven by
an E-727.3SDA digital piezo controller over USB.

The E-727 uses the same PI GCS2 DLL as the PI_N472 module, so the low-level
ccall wrappers (and the BOOL/TRUE/FALSE constants) are imported from PI_N472
instead of being duplicated. PI_N472 must therefore be included before this
module in HardwareImplementations.jl.

Axis handling: the number of axes is NOT hard-coded. On `initialize` the
controller is asked which axes it has (`SAI?`) and the first one or two are
used (or the ones given in `axes=`). The axes are mapped onto the generic
stage fields (x = first axis, y = second axis), so `gui(mirror)` reuses
StageInterface.gui1d / gui2d depending on `dimensions`.
=#

module PI_S331
    using ...MicroscopeControl.HardwareInterfaces.StageInterface

    import ...MicroscopeControl: export_state, initialize, shutdown

    import ..PI_N472: BOOL, TRUE, FALSE,
        PI_EnumerateUSB, PI_ConnectUSB, PI_IsConnected, PI_CloseConnection,
        PI_GetError, PI_GetInitError, PI_TranslateError, PI_qERR, PI_qIDN,
        PI_qSAI, PI_qSAI_ALL, PI_qPUN, PI_qTMN, PI_qTMX,
        PI_SVO, PI_qSVO, PI_MOV, PI_qMOV, PI_qPOS, PI_qONT, PI_HLT,
        PI_ATZ, PI_qATZ, PI_IsControllerReady, PI_qVOL

    include("types.jl")
    include("helper.jl")
    include("interface_methods.jl")

    export S331
    export move, getposition, getrange, home, stopmotion, servo, gui
    export moverelative, autozero, getvoltage, s331_printinfo
end
