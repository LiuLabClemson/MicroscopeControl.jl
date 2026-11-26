# Functions that every objective positioner inherits from the `Zpositioner` interface.

# Implementations extend these as `ObjPositionerInterface.initialize(...)` etc.
import ...MicroscopeControl: initialize, shutdown, export_state

# Extend Base.reset instead of defining a new `reset`, so exporting it does not
# clash with Base when users do `using MicroscopeControl`.
import Base: reset

"""
    move(positioner::Zpositioner, z::Float64)

Move the objective positioner to a specified position.

# Arguments
- `positioner::Zpositioner`: The objective positioner to be moved.
- `z::Float64`: The target z-coordinate for the positioner.
"""
function move(positioner::Zpositioner, z::Float64)
    @error "move() not implemented for $(typeof(positioner))"
end

"""
    get_position(positioner::Zpositioner)

Get the current position of the objective positioner.

# Arguments
- `positioner::Zpositioner`: The objective positioner whose position is to be retrieved.
"""
function get_position(positioner::Zpositioner)
    @error "get_position() not implemented for $(typeof(positioner))"
end

"""
    reset(positioner::Zpositioner)

Reset the objective positioner back to a safe "home" state.

# Arguments
- `positioner::Zpositioner`: The objective positioner to be reset.
"""
function reset(positioner::Zpositioner)
    @error "reset() not implemented for $(typeof(positioner))"
end

"""
    stop_motion(positioner::Zpositioner)

Stop the motion of the objective positioner.

# Arguments
- `positioner::Zpositioner`: The objective positioner whose motion will halt.
"""
function stop_motion(positioner::Zpositioner)
    @error "stop_motion() not implemented for $(typeof(positioner))"
end
