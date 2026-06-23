function loadlut(lut_path)  #TODO: Input LUT path
    #This works, now must load the LUT
    #lut_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT"
    loaded_lut = @ccall blink_sdk_path.Load_LUT_file(1::Cuint, lut_path::Ptr{UInt8})::Cint

end

#=
function writesingleimage(slm::MLSLM)
    single_image = slm.phase .* 255 #Converting 0 to 1 to scaled values between 0 and 255
    single_image = round.(Int, single_image) #Converting to integers
    single_image = convert(Array{UInt8}, single_image)

    board_number = 1
    wait_for_trigger::Cuint = 0
    flip_immediate::Cuint = 0
    output_pulse_image_flip::Cuint = 0
    output_pulse_image_refresh::Cuint = 0
    trigger_timout_ms::Cuint = 5000
    image_size::Cint = slm.height * slm.width

    #=
    single_image = reshape(single_image, 1, image_size)
    new_image = zeros(UInt8, image_size)    #Issues with Column vs Row major order
    new_image[:] .= single_image[1, :]
    =#
    new_image = reshape(single_image, image_size)
    #@ccall blink_sdk_path.Write_image(board_number::Cuint, 
    #    new_image::Ptr{Cint}, image_size::Cint, wait_for_trigger::Cuint, flip_immediate::Cuint, output_pulse_image_flip::Cuint, 
    #    output_pulse_image_refresh::Cuint, trigger_timout_ms::Cuint)::Cint


    #this here is og code
    #@ccall blink_sdk_path.Write_image(board_number::Cuint, 
    #    new_image::Ptr{Cint}, trigger_timout_ms::Cuint)::Cint
    @ccall blink_sdk_path.Write_image(board_number::Cuint,
        new_image::Ptr{Cuchar}, trigger_timout_ms::Cuint)::Cint
    

    
end 
=#
function writesingleimage(slm::MLSLM; board_number::Integer=1, trigger_timeout_ms::Integer=5000)
    single_image = slm.phase .* 255
    single_image = round.(UInt8, clamp.(single_image, 0, 255))  # clamp before cast, see note below
    image_size = slm.height * slm.width
    new_image = reshape(single_image, image_size)

    result = @ccall blink_sdk_path.Write_image(
        board_number::Cint,
        new_image::Ptr{Cuchar},
        trigger_timeout_ms::Cuint
    )::Cint

    if result != 1
        error("Write_image failed (board=$board_number). SDK returned $result.")
    end

    # Block until hardware is actually ready for the next DMA / actually displaying this image
    imagewritecomplete(slm; board_number=board_number, trigger_timeout_ms=trigger_timeout_ms)

    return result
end

function writesequence(slm::MLSLM)
    @error "Not implemented"
end

function selectsequenceimage(slm::MLSLM)
    @error "Not implemented"
end

function imagewritecomplete(slm::MLSLM; board_number::Integer=1, trigger_timeout_ms::Integer=5000)
    result = @ccall blink_sdk_path.ImageWriteComplete(
        board_number::Cint,
        trigger_timeout_ms::Cuint
    )::Cint

    if result != 1
        error("ImageWriteComplete failed (board=$board_number). SDK returned $result.")
    end
    return result
end