function initializesdk()
    # Initialize the SLM
    #First Create variables needed to create SDK
    bit_depth::Cuint = 12
    n_boards_found = Ref{Cuint}(0)
    constructed_ok = Ref{Cuint}(1)
    is_nematic_type::Cuint = 1
    ram_write_enable::Cuint = 1
    use_gpu::Cuint = 1
    max_transient_frames::Cuint = 10

    #Creating the SDK

    #@ccall blink_sdk_path.Create_SDK(bit_depth::Cuint, 
    #    n_boards_found::Ref{Cuint}, constructed_ok::Ref{Cuint}, is_nematic_type::Cuint, ram_write_enable::Cuint, use_gpu::Cuint, 
    #    max_transient_frames::Cuint, 0::Cuint)::Cvoid

    @ccall blink_sdk_path.Create_SDK(n_boards_found::Ref{Cuint}, constructed_ok::Ref{Cuint})::Cvoid

    println(n_boards_found[])
    println(constructed_ok[])
end

function closesdk()
    # Close the SLM
    @ccall blink_sdk_path.Delete_SDK()::Cvoid
end