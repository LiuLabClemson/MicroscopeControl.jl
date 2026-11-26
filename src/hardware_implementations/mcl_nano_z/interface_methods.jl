function initialize(stage::MCLNanoZStage)
    if stage.connectionstatus
        @warn "Stage already initialized"
        return stage.id
    end

    @info "Initializing MCL Nano-Z stage"
    handle = inithandle(stage)
    if handle <= 0
        @error "Stage failed to initialize"
        return handle
    end

    detectaxis!(stage)
    StageInterface.getrange(stage)
    readposition(stage)
    stage.targ_z = stage.real_z

    @info "MCL Nano-Z stage initialized" handle serial = getserialnumber(stage) axis = stage.axis range = stage.range_z position = stage.real_z
    return handle
end

function shutdown(stage::MCLNanoZStage)
    @info "Shutting down MCL Nano-Z stage"
    releasehandle(stage)
    return nothing
end

"""
    move(stage::MCLNanoZStage, z::Real)

Move the stage to `z` (microns). Returns the Madlib error name string.
"""
function StageInterface.move(stage::MCLNanoZStage, z::Real)
    @info "Z stage moving to position: $z"
    return writeposition(stage, z)
end

"""
    getposition(stage::MCLNanoZStage)

Read the current Z position (microns); also stored in `stage.real_z`.
"""
function StageInterface.getposition(stage::MCLNanoZStage)
    return readposition(stage)
end

"""
    getrange(stage::MCLNanoZStage)

Read the calibrated Z range (microns); also stored in `stage.range_z`.
"""
function StageInterface.getrange(stage::MCLNanoZStage)
    return getcalibration(stage)
end

function StageInterface.home(stage::MCLNanoZStage)
    @info "Homing Z stage to $(stage.homepos)"
    return writeposition(stage, stage.homepos)
end

function StageInterface.stopmotion(stage::MCLNanoZStage)
    @warn "stopmotion not supported by $(stage.stagelabel) (piezo moves complete immediately)"
    return nothing
end

"""
    export_state(stage::MCLNanoZStage)
"""
function export_state(stage::MCLNanoZStage)
    attributes = Dict{String,Any}(
        "stage_label" => stage.stagelabel,
        "units" => stage.units,
        "id" => stage.id,
        "serial" => stage.serial,
        "axis" => stage.axis,
        "dimensions" => stage.dimensions,
        "connected" => stage.connectionstatus,
        "position_z" => stage.real_z,
        "target_z" => stage.targ_z,
        "range_z" => collect(stage.range_z),
        "home_position" => stage.homepos,
    )

    data = nothing
    children = Dict{String,Any}()

    return attributes, data, children
end
