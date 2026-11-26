struct ProductInformation
    data::NTuple{11, UInt8}
end

function Base.getproperty(x::Ptr{ProductInformation}, f::Symbol)
    f === :axis_bitmap && return Ptr{Cuchar}(x + 0)
    f === :ADC_resolution && return Ptr{Cshort}(x + 1)
    f === :DAC_resolution && return Ptr{Cshort}(x + 3)
    f === :Product_id && return Ptr{Cshort}(x + 5)
    f === :FirmwareVersion && return Ptr{Cshort}(x + 7)
    f === :FirmwareProfile && return Ptr{Cshort}(x + 9)
    return getfield(x, f)
end

function Base.getproperty(x::ProductInformation, f::Symbol)
    r = Ref{ProductInformation}(x)
    ptr = Base.unsafe_convert(Ptr{ProductInformation}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{ProductInformation}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::ProductInformation, private::Bool = false)
    (:axis_bitmap, :ADC_resolution, :DAC_resolution, :Product_id, :FirmwareVersion, :FirmwareProfile, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

const bool = Cuchar

const MCL_SUCCESS = 0

const MCL_GENERAL_ERROR = -1

const MCL_DEV_ERROR = -2

const MCL_DEV_NOT_ATTACHED = -3

const MCL_USAGE_ERROR = -4

const MCL_DEV_NOT_READY = -5

const MCL_ARGUMENT_ERROR = -6

const MCL_INVALID_AXIS = -7

const MCL_INVALID_HANDLE = -8

# C header macro, not valid Julia (would throw UndefVarError at load time)
# const MADLIB_API = __declspec(dllimport)

