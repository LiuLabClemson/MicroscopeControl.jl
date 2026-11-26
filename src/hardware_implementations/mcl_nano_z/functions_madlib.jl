function MCL_DLLVersion(version, revision)
    ccall((:MCL_DLLVersion, MadLib), Cvoid, (Ptr{Cshort}, Ptr{Cshort}), version, revision)
end

# no prototype is found for this function at Madlib.h:35:18, please use with caution
function MCL_InitHandle()
    ccall((:MCL_InitHandle, MadLib), Cint, ())
end

function MCL_GrabHandle(device)
    ccall((:MCL_GrabHandle, MadLib), Cint, (Cshort,), device)
end

# no prototype is found for this function at Madlib.h:37:18, please use with caution
function MCL_InitHandleOrGetExisting()
    ccall((:MCL_InitHandleOrGetExisting, MadLib), Cint, ())
end

function MCL_GrabHandleOrGetExisting(device)
    ccall((:MCL_GrabHandleOrGetExisting, MadLib), Cint, (Cshort,), device)
end

function MCL_GetHandleBySerial(serial)
    ccall((:MCL_GetHandleBySerial, MadLib), Cint, (Cshort,), serial)
end

# no prototype is found for this function at Madlib.h:40:18, please use with caution
function MCL_GrabAllHandles()
    ccall((:MCL_GrabAllHandles, MadLib), Cint, ())
end

function MCL_GetAllHandles(handles, size)
    ccall((:MCL_GetAllHandles, MadLib), Cint, (Ptr{Cint}, Cint), handles, size)
end

# no prototype is found for this function at Madlib.h:42:18, please use with caution
function MCL_NumberOfCurrentHandles()
    ccall((:MCL_NumberOfCurrentHandles, MadLib), Cint, ())
end

function MCL_ReleaseHandle(handle)
    ccall((:MCL_ReleaseHandle, MadLib), Cvoid, (Cint,), handle)
end

# no prototype is found for this function at Madlib.h:44:18, please use with caution
function MCL_ReleaseAllHandles()
    ccall((:MCL_ReleaseAllHandles, MadLib), Cvoid, ())
end

function MCL_SingleReadZ(handle)
    ccall((:MCL_SingleReadZ, MadLib), Cdouble, (Cint,), handle)
end

function MCL_SingleReadN(axis, handle)
    ccall((:MCL_SingleReadN, MadLib), Cdouble, (Cuint, Cint), axis, handle)
end

function MCL_SingleWriteZ(position, handle)
    ccall((:MCL_SingleWriteZ, MadLib), Cint, (Cdouble, Cint), position, handle)
end

function MCL_SingleWriteN(position, axis, handle)
    ccall((:MCL_SingleWriteN, MadLib), Cint, (Cdouble, Cuint, Cint), position, axis, handle)
end

function MCL_MonitorZ(position, handle)
    ccall((:MCL_MonitorZ, MadLib), Cdouble, (Cdouble, Cint), position, handle)
end

function MCL_MonitorN(position, axis, handle)
    ccall((:MCL_MonitorN, MadLib), Cdouble, (Cdouble, Cuint, Cint), position, axis, handle)
end

function MCL_ReadEncoderZ(handle)
    ccall((:MCL_ReadEncoderZ, MadLib), Cdouble, (Cint,), handle)
end

function MCL_ResetEncoderZ(handle)
    ccall((:MCL_ResetEncoderZ, MadLib), Cint, (Cint,), handle)
end

function MCL_ThetaX(milliradians, actual, handle)
    ccall((:MCL_ThetaX, MadLib), Cint, (Cdouble, Ptr{Cdouble}, Cint), milliradians, actual, handle)
end

function MCL_ThetaY(milliradians, actual, handle)
    ccall((:MCL_ThetaY, MadLib), Cint, (Cdouble, Ptr{Cdouble}, Cint), milliradians, actual, handle)
end

function MCL_MoveZCenter(position, actual, handle)
    ccall((:MCL_MoveZCenter, MadLib), Cint, (Cdouble, Ptr{Cdouble}, Cint), position, actual, handle)
end

function MCL_LevelZ(position, handle)
    ccall((:MCL_LevelZ, MadLib), Cint, (Cdouble, Cint), position, handle)
end

function MCL_CFocusSetFocusMode(focusModeOn, handle)
    ccall((:MCL_CFocusSetFocusMode, MadLib), Cint, (bool, Cint), focusModeOn, handle)
end

function MCL_CFocusStep(relativePositionChange, handle)
    ccall((:MCL_CFocusStep, MadLib), Cint, (Cdouble, Cint), relativePositionChange, handle)
end

function MCL_CFocusGetFocusMode(focusLocked, handle)
    ccall((:MCL_CFocusGetFocusMode, MadLib), Cint, (Ptr{Cint}, Cint), focusLocked, handle)
end

function MCL_ReadWaveFormN(axis, DataPoints, milliseconds, waveform, handle)
    ccall((:MCL_ReadWaveFormN, MadLib), Cint, (Cuint, Cuint, Cdouble, Ptr{Cdouble}, Cint), axis, DataPoints, milliseconds, waveform, handle)
end

function MCL_Setup_ReadWaveFormN(axis, DataPoints, milliseconds, handle)
    ccall((:MCL_Setup_ReadWaveFormN, MadLib), Cint, (Cuint, Cuint, Cdouble, Cint), axis, DataPoints, milliseconds, handle)
end

function MCL_Trigger_ReadWaveFormN(axis, DataPoints, waveform, handle)
    ccall((:MCL_Trigger_ReadWaveFormN, MadLib), Cint, (Cuint, Cuint, Ptr{Cdouble}, Cint), axis, DataPoints, waveform, handle)
end

function MCL_LoadWaveFormN(axis, DataPoints, milliseconds, waveform, handle)
    ccall((:MCL_LoadWaveFormN, MadLib), Cint, (Cuint, Cuint, Cdouble, Ptr{Cdouble}, Cint), axis, DataPoints, milliseconds, waveform, handle)
end

function MCL_Setup_LoadWaveFormN(axis, DataPoints, milliseconds, waveform, handle)
    ccall((:MCL_Setup_LoadWaveFormN, MadLib), Cint, (Cuint, Cuint, Cdouble, Ptr{Cdouble}, Cint), axis, DataPoints, milliseconds, waveform, handle)
end

function MCL_Trigger_LoadWaveFormN(axis, handle)
    ccall((:MCL_Trigger_LoadWaveFormN, MadLib), Cint, (Cuint, Cint), axis, handle)
end

function MCL_TriggerWaveformAcquisition(axis, DataPoints, waveform, handle)
    ccall((:MCL_TriggerWaveformAcquisition, MadLib), Cint, (Cuint, Cuint, Ptr{Cdouble}, Cint), axis, DataPoints, waveform, handle)
end

function MCL_WfmaSetup(wfDacX, wfDacY, wfDacZ, dataPointsPerAxis, milliseconds, iterations, handle)
    ccall((:MCL_WfmaSetup, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble, Cushort, Cint), wfDacX, wfDacY, wfDacZ, dataPointsPerAxis, milliseconds, iterations, handle)
end

function MCL_WfmaTriggerAndRead(wfAdcX, wfAdcY, wfAdcZ, handle)
    ccall((:MCL_WfmaTriggerAndRead, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), wfAdcX, wfAdcY, wfAdcZ, handle)
end

function MCL_WfmaTrigger(handle)
    ccall((:MCL_WfmaTrigger, MadLib), Cint, (Cint,), handle)
end

function MCL_WfmaRead(wfAdcX, wfAdcY, wfAdcZ, handle)
    ccall((:MCL_WfmaRead, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), wfAdcX, wfAdcY, wfAdcZ, handle)
end

function MCL_WfmaStop(handle)
    ccall((:MCL_WfmaStop, MadLib), Cint, (Cint,), handle)
end

function MCL_IssBindClockToAxis(clock, mode, axis, handle)
    ccall((:MCL_IssBindClockToAxis, MadLib), Cint, (Cint, Cint, Cint, Cint), clock, mode, axis, handle)
end

function MCL_IssConfigurePolarity(clock, mode, handle)
    ccall((:MCL_IssConfigurePolarity, MadLib), Cint, (Cint, Cint, Cint), clock, mode, handle)
end

function MCL_IssSetClock(clock, mode, handle)
    ccall((:MCL_IssSetClock, MadLib), Cint, (Cint, Cint, Cint), clock, mode, handle)
end

function MCL_IssResetDefaults(handle)
    ccall((:MCL_IssResetDefaults, MadLib), Cint, (Cint,), handle)
end

function MCL_ChangeClock(milliseconds, clock, handle)
    ccall((:MCL_ChangeClock, MadLib), Cint, (Cdouble, Cshort, Cint), milliseconds, clock, handle)
end

function MCL_PixelClock(handle)
    ccall((:MCL_PixelClock, MadLib), Cint, (Cint,), handle)
end

function MCL_LineClock(handle)
    ccall((:MCL_LineClock, MadLib), Cint, (Cint,), handle)
end

function MCL_FrameClock(handle)
    ccall((:MCL_FrameClock, MadLib), Cint, (Cint,), handle)
end

function MCL_AuxClock(handle)
    ccall((:MCL_AuxClock, MadLib), Cint, (Cint,), handle)
end

function MCL_GetClockFrequency(adcfreq, dacfreq, handle)
    ccall((:MCL_GetClockFrequency, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), adcfreq, dacfreq, handle)
end

function MCL_SequenceLoad(axis, sequence, seqSize, handle)
    ccall((:MCL_SequenceLoad, MadLib), Cint, (Cint, Ptr{Cdouble}, Cint, Cint), axis, sequence, seqSize, handle)
end

function MCL_SequenceClear(handle)
    ccall((:MCL_SequenceClear, MadLib), Cint, (Cint,), handle)
end

function MCL_SequenceStart(handle)
    ccall((:MCL_SequenceStart, MadLib), Cint, (Cint,), handle)
end

function MCL_SequenceStop(handle)
    ccall((:MCL_SequenceStop, MadLib), Cint, (Cint,), handle)
end

function MCL_SequenceGetMax(max, handle)
    ccall((:MCL_SequenceGetMax, MadLib), Cint, (Ptr{Cint}, Cint), max, handle)
end

function MCL_GetCalibration(axis, handle)
    ccall((:MCL_GetCalibration, MadLib), Cdouble, (Cuint, Cint), axis, handle)
end

function MCL_TipTiltHeight(handle)
    ccall((:MCL_TipTiltHeight, MadLib), Cdouble, (Cint,), handle)
end

function MCL_TipTiltWidth(handle)
    ccall((:MCL_TipTiltWidth, MadLib), Cdouble, (Cint,), handle)
end

function MCL_MinMaxThetaX(min, max, handle)
    ccall((:MCL_MinMaxThetaX, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), min, max, handle)
end

function MCL_MinMaxThetaY(min, max, handle)
    ccall((:MCL_MinMaxThetaY, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), min, max, handle)
end

function MCL_GetTipTiltThetaX(handle)
    ccall((:MCL_GetTipTiltThetaX, MadLib), Cdouble, (Cint,), handle)
end

function MCL_GetTipTiltThetaY(handle)
    ccall((:MCL_GetTipTiltThetaY, MadLib), Cdouble, (Cint,), handle)
end

function MCL_GetTipTiltCenter(handle)
    ccall((:MCL_GetTipTiltCenter, MadLib), Cdouble, (Cint,), handle)
end

function MCL_CurrentMinMaxThetaX(min, max, handle)
    ccall((:MCL_CurrentMinMaxThetaX, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), min, max, handle)
end

function MCL_CurrentMinMaxThetaY(min, max, handle)
    ccall((:MCL_CurrentMinMaxThetaY, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), min, max, handle)
end

function MCL_CurrentMinMaxCenter(min, max, handle)
    ccall((:MCL_CurrentMinMaxCenter, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Cint), min, max, handle)
end

function MCL_GetFirmwareVersion(version, profile, handle)
    ccall((:MCL_GetFirmwareVersion, MadLib), Cint, (Ptr{Cshort}, Ptr{Cshort}, Cint), version, profile, handle)
end

function MCL_GetSerialNumber(handle)
    ccall((:MCL_GetSerialNumber, MadLib), Cint, (Cint,), handle)
end

function MCL_GetProductInfo(pi, handle)
    ccall((:MCL_GetProductInfo, MadLib), Cint, (Ptr{ProductInformation}, Cint), pi, handle)
end

function MCL_PrintDeviceInfo(handle)
    ccall((:MCL_PrintDeviceInfo, MadLib), Cvoid, (Cint,), handle)
end

function MCL_DeviceAttached(milliseconds, handle)
    ccall((:MCL_DeviceAttached, MadLib), bool, (Cint, Cint), milliseconds, handle)
end

# no prototype is found for this function at Madlib.h:109:21, please use with caution
function MCL_CorrectDriverVersion()
    ccall((:MCL_CorrectDriverVersion, MadLib), bool, ())
end

function MCL_GetCommandedPosition(xCom, yCom, zCom, handle)
    ccall((:MCL_GetCommandedPosition, MadLib), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), xCom, yCom, zCom, handle)
end

