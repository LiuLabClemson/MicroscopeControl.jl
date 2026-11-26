#=
Helper functions wrapping the Madlib calls from functions_madlib.jl for MCLNanoZStage.
=#

const MCL_ERRORS = Dict{Int,String}(
    MCL_SUCCESS => "MCL_SUCCESS",
    MCL_GENERAL_ERROR => "MCL_GENERAL_ERROR",
    MCL_DEV_ERROR => "MCL_DEV_ERROR",
    MCL_DEV_NOT_ATTACHED => "MCL_DEV_NOT_ATTACHED",
    MCL_USAGE_ERROR => "MCL_USAGE_ERROR",
    MCL_DEV_NOT_READY => "MCL_DEV_NOT_READY",
    MCL_ARGUMENT_ERROR => "MCL_ARGUMENT_ERROR",
    MCL_INVALID_AXIS => "MCL_INVALID_AXIS",
    MCL_INVALID_HANDLE => "MCL_INVALID_HANDLE",
)

errstring(code::Integer) = get(MCL_ERRORS, Int(code), "MCL_UNKNOWN_ERROR($code)")

"""
    checkerror(code, fname)

Log an error if `code` is not `MCL_SUCCESS`. Returns the error name string.
"""
function checkerror(code::Integer, fname::AbstractString)
    if code != MCL_SUCCESS
        @error "$fname failed: $(errstring(code))"
    end
    return errstring(code)
end

# Madlib functions returning a double (position, calibration) return a negative
# integer error code on failure.
isreturnerror(val::Float64) = val < 0 && isinteger(val) && haskey(MCL_ERRORS, Int(val))

# ---------------------------------------------------------------- handles

"""
    currenthandles()

Return the handles currently held by this process.
"""
function currenthandles()
    n = MCL_NumberOfCurrentHandles()
    n <= 0 && return Cint[]
    buf = zeros(Cint, n)
    count = MCL_GetAllHandles(buf, Cint(n))
    return buf[1:max(count, 0)]
end

"""
    inithandle(stage::MCLNanoZStage)

Acquire a device handle. If `stage.serial == 0` the first available device is
grabbed, otherwise the device with the matching serial number. Handles grabbed
along the way that belong to other devices are released again, so other MCL
stages (e.g. `MCLStage`) can still be initialized in the same session.
"""
function inithandle(stage::MCLNanoZStage)
    if stage.serial == 0
        handle = MCL_InitHandle()
    else
        before = currenthandles()
        MCL_GrabAllHandles()
        handle = MCL_GetHandleBySerial(Cshort(stage.serial))
        for h in currenthandles()
            if h != handle && !(h in before)
                MCL_ReleaseHandle(h)
            end
        end
    end

    stage.id = handle
    stage.connectionstatus = handle > 0
    if handle <= 0
        @error "MCL Nano-Z stage not found" serial = stage.serial
    end
    return handle
end

"""
    releasehandle(stage::MCLNanoZStage)

Release the device handle and mark the stage as disconnected.
"""
function releasehandle(stage::MCLNanoZStage)
    if stage.id > 0
        MCL_ReleaseHandle(stage.id)
    end
    stage.id = 0
    stage.connectionstatus = false
    return nothing
end

# ---------------------------------------------------------------- device information

"""
    nanoz_isattached(stage::MCLNanoZStage; timeout_ms=500)

Check whether the device is still attached (waits up to `timeout_ms`).
"""
function nanoz_isattached(stage::MCLNanoZStage; timeout_ms::Integer=500)
    attached = MCL_DeviceAttached(Cint(timeout_ms), stage.id) != 0
    stage.connectionstatus = attached
    attached || @error "MCL Nano-Z stage not attached"
    return attached
end

"""
    nanoz_printinfo(stage::MCLNanoZStage)

Print the device information reported by Madlib.
"""
function nanoz_printinfo(stage::MCLNanoZStage)
    MCL_PrintDeviceInfo(stage.id)
    return nothing
end

getserialnumber(stage::MCLNanoZStage) = Int(MCL_GetSerialNumber(stage.id))

"""
    getproductinfo(stage::MCLNanoZStage)

Return the `ProductInformation` struct, or `nothing` on failure.
"""
function getproductinfo(stage::MCLNanoZStage)
    info = Ref(ProductInformation(ntuple(_ -> 0x00, 11)))
    err = MCL_GetProductInfo(info, stage.id)
    if err != MCL_SUCCESS
        checkerror(err, "MCL_GetProductInfo")
        return nothing
    end
    return info[]
end

"""
    detectaxis!(stage::MCLNanoZStage)

Set `stage.axis` to the axis that is valid on this controller, based on the
product info axis bitmap (bit 0 = X, bit 1 = Y, bit 2 = Z, bit 3 = AUX).
Z is preferred when present. Keeps the current value if detection fails.
"""
function detectaxis!(stage::MCLNanoZStage)
    info = getproductinfo(stage)
    if info === nothing
        @warn "Could not read product info, using axis $(stage.axis)"
        return stage.axis
    end
    bitmap = info.axis_bitmap
    for axis in (3, 1, 2, 4)
        if (bitmap >> (axis - 1)) & 0x01 == 0x01
            stage.axis = axis
            return axis
        end
    end
    @warn "No valid axis in product info (bitmap = $bitmap), using axis $(stage.axis)"
    return stage.axis
end

"""
    getcalibration(stage::MCLNanoZStage)

Read the calibrated travel range of the axis and store it in `stage.range_z`.
"""
function getcalibration(stage::MCLNanoZStage)
    cal = MCL_GetCalibration(Cuint(stage.axis), stage.id)
    if isreturnerror(cal)
        checkerror(Int(cal), "MCL_GetCalibration")
        return stage.range_z
    end
    stage.range_z = (0.0, cal)
    return stage.range_z
end

# ---------------------------------------------------------------- movement

"""
    readposition(stage::MCLNanoZStage)

Read the current position and store it in `stage.real_z`.
Returns the position, or `nothing` on error.
"""
function readposition(stage::MCLNanoZStage)
    pos = MCL_SingleReadN(Cuint(stage.axis), stage.id)
    if isreturnerror(pos)
        checkerror(Int(pos), "MCL_SingleReadN")
        return nothing
    end
    stage.real_z = pos
    return pos
end

"""
    writeposition(stage::MCLNanoZStage, position)

Command the stage to `position` (microns). Returns the Madlib error name string.
"""
function writeposition(stage::MCLNanoZStage, position::Real)
    lo, hi = stage.range_z
    if position < lo || position > hi
        @error "Target position $position out of range $(stage.range_z)"
        return errstring(MCL_ARGUMENT_ERROR)
    end
    err = MCL_SingleWriteN(Cdouble(position), Cuint(stage.axis), stage.id)
    if err == MCL_SUCCESS
        stage.targ_z = position
    end
    return checkerror(err, "MCL_SingleWriteN")
end

"""
    nanoz_monitor(stage::MCLNanoZStage, position)

Move to `position` (microns) and read the position back after the move.
Returns the read position, or `nothing` on error.
"""
function nanoz_monitor(stage::MCLNanoZStage, position::Real)
    lo, hi = stage.range_z
    if position < lo || position > hi
        @error "Target position $position out of range $(stage.range_z)"
        return nothing
    end
    pos = MCL_MonitorN(Cdouble(position), Cuint(stage.axis), stage.id)
    if isreturnerror(pos)
        checkerror(Int(pos), "MCL_MonitorN")
        return nothing
    end
    stage.targ_z = position
    stage.real_z = pos
    return pos
end
