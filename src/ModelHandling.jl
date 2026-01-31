# module _ModelHandling


# =============================================================================
# ModelManager: Unified Model Interface for Change Point Detection
# =============================================================================

"""
`ModelManager{T<:AbstractModelSpec}`

A wrapper around any model specification (ODE, Difference, or Regression) that provides
a unified interface to:

- Access the initial condition
- Segment model-specific data for a given interval
- Generate per-segment models with updated parameters
- Identify the model type for dispatching logic

This allows the objective function and other algorithms to remain model-agnostic.
"""
struct ModelManager{T<:AbstractModelSpec}
    base_model::T
end

# =============================================================================
# Initial Condition Accessor
# =============================================================================

"""
`get_initial_condition(manager::ModelManager) -> Any`

Returns the initial condition used by the model. For:

- ODE models: returns the vector of initial states.
- Difference models: returns the scalar initial value.
- Regression models: returns `nothing` (no initial condition concept).
"""
get_initial_condition(manager::ModelManager{ODEModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{DifferenceModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{RegressionModelSpec}) = nothing

# =============================================================================
# Initial Condition Updating
# =============================================================================

"""
`update_initial_condition(manager::ModelManager, sim_data::DataFrame)`

Returns the updated initial condition for the next segment based on
the output of the last segment's simulation.
"""
function update_initial_condition(manager::ModelManager{ODEModelSpec}, sim_data)
    return sim_data[:,end]
end

function update_initial_condition(manager::ModelManager{DifferenceModelSpec}, sim_data)
    #@show sim_data
    return sim_data[end]
end

function update_initial_condition(manager::ModelManager{RegressionModelSpec}, sim_data)
    return nothing
end


# =============================================================================
# Model Segmentation
# =============================================================================

"""
`segment_model(manager, seg_pars, parnames, idx_start, idx_end, u0) -> AbstractModelSpec`

Builds a new per-segment model specification using:

- The segment-specific parameters `seg_pars` and their names `parnames`
- The index range `[idx_start:idx_end]` defining the segment
- The initial condition `u0` passed from the last segment

Dispatches based on model type to slice and prepare data correctly.
"""
function segment_model(
    manager::ModelManager{ODEModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0
    )

    model = manager.base_model
    
    return ODEModelSpec(model.model_function, pars, u0, (idx_start, idx_end))
end

function segment_model(
    manager::ModelManager{DifferenceModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0
)
    model = manager.base_model
    
    segmented_extra = map(x -> x[idx_start:idx_end], model.extra_data)
    num_steps = idx_end - idx_start + 1
    return DifferenceModelSpec(model.model_function, pars, u0, num_steps, segmented_extra)
end

function segment_model(
    manager::ModelManager{RegressionModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    _  # no initial condition
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return RegressionModelSpec(model.model_function, pars, time_steps)
end

# =============================================================================
# Optional Model Type Name
# =============================================================================

"""
`get_model_type(manager::ModelManager) -> String`

Returns a string identifier for the model type: "ODE", "Difference", or "Regression".
Useful for logging or conditional logic.
"""
get_model_type(manager::ModelManager{ODEModelSpec}) = "ODE"
get_model_type(manager::ModelManager{DifferenceModelSpec}) = "Difference"
get_model_type(manager::ModelManager{RegressionModelSpec}) = "Regression"

#end # module
