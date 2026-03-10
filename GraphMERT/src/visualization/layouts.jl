"""
Layout algorithms for GraphMERT knowledge graph visualization.

Provides various graph layout algorithms optimized for knowledge graphs.
"""

using Graphs

# Check availability of NetworkLayout
let
    global NETWORKLAYOUT_AVAILABLE = false
    try
        using NetworkLayout
        NETWORKLAYOUT_AVAILABLE = true
    catch
        # Package not available
    end
end

"""
    compute_layout(mg::MetaGraph, algorithm::Symbol=:spring; kwargs...)

Compute node positions for graph visualization using specified layout algorithm.

# Arguments
- `mg::MetaGraph`: Graph to layout
- `algorithm::Symbol`: Layout algorithm (:spring, :circular, :random, :stress, :shell, :bipartite)
- `kwargs...`: Algorithm-specific parameters

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions as (x, y) coordinates

# Examples
```julia
mg = kg_to_graphs_format(kg)
positions = compute_layout(mg, :spring, C=2.0, iterations=100)
```
"""
function compute_layout(mg::MetaGraph, algorithm::Symbol=:spring; kwargs...)
    n = nv(mg)

    if algorithm == :circular
        return circular_layout(n; kwargs...)
    elseif algorithm == :random
        return random_layout(n; kwargs...)
    elseif algorithm == :spring
        return spring_layout(mg; kwargs...)
    elseif algorithm == :stress
        return stress_layout(mg; kwargs...)
    elseif algorithm == :shell
        return shell_layout(mg; kwargs...)
    elseif algorithm == :bipartite
        return bipartite_layout(mg; kwargs...)
    else
        @warn "Unknown layout algorithm: $algorithm, using spring layout"
        return spring_layout(mg; kwargs...)
    end
end

# Layout algorithm implementations

"""
    circular_layout(n::Int; radius::Float64=1.0)

Arrange nodes in a circle.

# Arguments
- `n::Int`: Number of nodes
- `radius::Float64`: Circle radius

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function circular_layout(n::Int; radius::Float64=1.0)
    positions = Vector{Tuple{Float64, Float64}}(undef, n)

    if n == 1
        return [(0.0, 0.0)]
    end

    for i in 1:n
        angle = 2π * (i - 1) / n
        x = radius * cos(angle)
        y = radius * sin(angle)
        positions[i] = (x, y)
    end

    return positions
end

"""
    random_layout(n::Int; scale::Float64=1.0)

Arrange nodes randomly in a square.

# Arguments
- `n::Int`: Number of nodes
- `scale::Float64`: Scale factor for positions

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function random_layout(n::Int; scale::Float64=1.0)
    positions = Vector{Tuple{Float64, Float64}}(undef, n)

    for i in 1:n
        x = scale * (2 * rand() - 1)
        y = scale * (2 * rand() - 1)
        positions[i] = (x, y)
    end

    return positions
end

"""
    spring_layout(mg::MetaGraph; C::Float64=2.0, iterations::Int=100, initial_temp::Float64=2.0)

Force-directed spring layout algorithm.

# Arguments
- `mg::MetaGraph`: Graph to layout
- `C::Float64`: Optimal distance between nodes
- `iterations::Int`: Number of iterations
- `initial_temp::Float64`: Initial temperature for annealing

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function spring_layout(mg::MetaGraph; C::Float64=2.0, iterations::Int=100, initial_temp::Float64=2.0)
    n = nv(mg)

    if NETWORKLAYOUT_AVAILABLE
        # Use NetworkLayout.jl if available
        try
            layout_func = NetworkLayout.Spring(iterations=iterations, C=C)
            positions_2d = layout_func(mg)
            return [(positions_2d[1,i], positions_2d[2,i]) for i in 1:n]
        catch
            @warn "NetworkLayout.jl failed, falling back to simple implementation"
        end
    end

    # Simple spring layout implementation
    positions = [(2*rand()-1, 2*rand()-1) for _ in 1:n]
    temp = initial_temp

    for _ in 1:iterations
        # Calculate repulsive forces
        forces = [(0.0, 0.0) for _ in 1:n]

        for i in 1:n
            for j in 1:n
                if i != j
                    dx = positions[i][1] - positions[j][1]
                    dy = positions[i][2] - positions[j][2]
                    dist = sqrt(dx^2 + dy^2)

                    if dist > 0
                        # Repulsive force
                        force = C^2 / dist
                        forces[i] = (forces[i][1] + force * dx / dist,
                                   forces[i][2] + force * dy / dist)
                    end
                end
            end
        end

        # Calculate attractive forces for edges
        for e in edges(mg)
            i, j = src(e), dst(e)
            dx = positions[i][1] - positions[j][1]
            dy = positions[i][2] - positions[j][2]
            dist = sqrt(dx^2 + dy^2)

            if dist > 0
                force = dist^2 / C
                forces[i] = (forces[i][1] - force * dx / dist,
                           forces[i][2] - force * dy / dist)
                forces[j] = (forces[j][1] + force * dx / dist,
                           forces[j][2] + force * dy / dist)
            end
        end

        # Update positions
        for i in 1:n
            fx, fy = forces[i]
            scale = min(temp, sqrt(fx^2 + fy^2))
            if scale > 0
                fx, fy = scale * fx / sqrt(fx^2 + fy^2), scale * fy / sqrt(fx^2 + fy^2)
            end

            positions[i] = (positions[i][1] + fx, positions[i][2] + fy)
        end

        temp *= 0.95  # Cool down
    end

    return positions
end

"""
    stress_layout(mg::MetaGraph; iterations::Int=50, tolerance::Float64=1e-4)

Stress majorization layout algorithm for graph drawing.

# Arguments
- `mg::MetaGraph`: Graph to layout
- `iterations::Int`: Maximum iterations
- `tolerance::Float64`: Convergence tolerance

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function stress_layout(mg::MetaGraph; iterations::Int=50, tolerance::Float64=1e-4)
    n = nv(mg)

    if NETWORKLAYOUT_AVAILABLE
        try
            layout_func = NetworkLayout.Stress(iterations=iterations)
            positions_2d = layout_func(mg)
            return [(positions_2d[1,i], positions_2d[2,i]) for i in 1:n]
        catch
            @warn "NetworkLayout.jl failed, falling back to circular layout"
        end
    end

    # Fallback to circular layout
    return circular_layout(n)
end

"""
    shell_layout(mg::MetaGraph; shells::Union{Nothing,Vector{Vector{Int}}}=nothing)

Shell layout arranging nodes in concentric circles.

# Arguments
- `mg::MetaGraph`: Graph to layout
- `shells::Union{Nothing,Vector{Vector{Int}}}`: Node indices for each shell, or auto-detected

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function shell_layout(mg::MetaGraph; shells::Union{Nothing,Vector{Vector{Int}}}=nothing)
    n = nv(mg)

    if shells === nothing
        # Auto-detect shells using BFS from highest degree node
        degrees = [degree(mg, v) for v in vertices(mg)]
        start_node = argmax(degrees)

        visited = falses(n)
        shells = Vector{Vector{Int}}()

        current_shell = [start_node]
        visited[start_node] = true

        while !all(visited)
            push!(shells, current_shell)
            next_shell = Int[]

            for node in current_shell
                for neighbor in neighbors(mg, node)
                    if !visited[neighbor]
                        push!(next_shell, neighbor)
                        visited[neighbor] = true
                    end
                end
            end

            if isempty(next_shell)
                # Add remaining unvisited nodes to last shell
                for i in 1:n
                    if !visited[i]
                        push!(next_shell, i)
                        visited[i] = true
                    end
                end
            end

            current_shell = next_shell
        end

        push!(shells, current_shell)
    end

    positions = Vector{Tuple{Float64, Float64}}(undef, n)

    for (shell_idx, shell_nodes) in enumerate(shells)
        radius = shell_idx * 0.5
        n_shell = length(shell_nodes)

        for (i, node) in enumerate(shell_nodes)
            angle = 2π * (i - 1) / n_shell
            x = radius * cos(angle)
            y = radius * sin(angle)
            positions[node] = (x, y)
        end
    end

    return positions
end

"""
    bipartite_layout(mg::MetaGraph; partition::Union{Nothing,Vector{Int}}=nothing)

Bipartite layout arranging nodes in two columns.

# Arguments
- `mg::MetaGraph`: Graph to layout (assumed bipartite)
- `partition::Union{Nothing,Vector{Int}}`: Node partition, or auto-detected

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function bipartite_layout(mg::MetaGraph; partition::Union{Nothing,Vector{Int}}=nothing)
    n = nv(mg)

    if partition === nothing
        # Simple heuristic: partition by degree
        degrees = [degree(mg, v) for v in vertices(mg)]
        median_degree = median(degrees)
        partition = [d > median_degree ? 2 : 1 for d in degrees]
    end

    positions = Vector{Tuple{Float64, Float64}}(undef, n)

    left_nodes = findall(==(1), partition)
    right_nodes = findall(==(2), partition)

    # Position left nodes
    left_y = range(-1, 1, length=length(left_nodes))
    for (i, node) in enumerate(left_nodes)
        positions[node] = (-1.0, left_y[i])
    end

    # Position right nodes
    right_y = range(-1, 1, length=length(right_nodes))
    for (i, node) in enumerate(right_nodes)
        positions[node] = (1.0, right_y[i])
    end

    return positions
end

"""
    hierarchical_layout(mg::MetaGraph; root::Union{Nothing,Int}=nothing)

Hierarchical layout for directed acyclic graphs.

# Arguments
- `mg::MetaGraph`: Graph to layout
- `root::Union{Nothing,Int}`: Root node, or auto-detected

# Returns
- `Vector{Tuple{Float64, Float64}}`: Node positions
"""
function hierarchical_layout(mg::MetaGraph; root::Union{Nothing,Int}=nothing)
    n = nv(mg)

    if root === nothing
        # Find node with highest out-degree as root
        out_degrees = [length(outneighbors(mg, v)) for v in vertices(mg)]
        root = argmax(out_degrees)
    end

    # Perform topological sort if possible, otherwise use BFS levels
    levels = Vector{Vector{Int}}()
    visited = falses(n)
    queue = [root]
    visited[root] = true

    current_level = [root]

    while !isempty(queue)
        push!(levels, current_level)
        next_level = Int[]

        for node in current_level
            for neighbor in outneighbors(mg, node)
                if !visited[neighbor]
                    push!(next_level, neighbor)
                    visited[neighbor] = true
                end
            end
        end

        current_level = next_level
        queue = next_level
    end

    # Add remaining nodes to last level
    remaining = [v for v in vertices(mg) if !visited[v]]
    if !isempty(remaining)
        push!(levels, remaining)
        for v in remaining
            visited[v] = true
        end
    end

    positions = Vector{Tuple{Float64, Float64}}(undef, n)

    for (level_idx, level_nodes) in enumerate(levels)
        x = (level_idx - 1) * 0.5
        level_y = range(-1, 1, length=length(level_nodes))

        for (i, node) in enumerate(level_nodes)
            positions[node] = (x, level_y[i])
        end
    end

    return positions
end
