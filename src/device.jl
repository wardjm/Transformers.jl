using Adapt
using Preferences
using Functors
using MLDataDevices
using MLDataDevices: AbstractDevice, CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice

is_precompiling() = ccall(:jl_generating_output, Cint, ()) == 1

"""
    gpu_backend()

The name of the currently selected gpu backend, as set by `MLDataDevices.gpu_backend!`.
 Returns `""` when no backend preference is set, in which case the backend is auto-detected.
"""
gpu_backend() = something(Preferences.load_preference(MLDataDevices, "gpu_backend", nothing), "")

"""
    enable_gpu(t=true)

Enable gpu for `todevice`, disable with `enable_gpu(false)`. The backend is selected by
 `MLDataDevices.gpu_backend!`, or auto-detected when no preference is set. Should only be
 used in user scripts.
"""
function enable_gpu(t::Bool=true)
    if is_precompiling()
        @error """
        `Transformers.enable_gpu` is called during precompilation, result in a no-op.
        `enable_gpu` should only be used in user scripts.
        """
        return todevice
    end
    if !t
        return @eval @inline todevice(args...; kws...) = tocpudevice(args...; kws...)
    end
    backend = gpu_backend()
    if backend == "CUDA"
        @eval Main using CUDA
    elseif backend == "AMDGPU"
        @eval Main using AMDGPU
    elseif backend == "Metal"
        @eval Main using Metal
    elseif backend == "CPU" || isempty(backend)
        # no trigger package to load; `gpu_device` picks whatever is functional
    else
        error("Unsupported GPU backend: $backend")
    end
    MLDataDevices.reset_gpu_device!()
    MLDataDevices.gpu_device(; force = backend != "CPU")
    return @eval @inline todevice(args...; kws...) = togpudevice(args...; kws...)
end

"""
    todevice(x)

Move data to device, only when gpu is enable with `enable_gpu`, basically equal
 `MLDataDevices.gpu_device()(x)`. Otherwise just `MLDataDevices.cpu_device()(x)`.
"""
@inline todevice(args...; kws...) = tocpudevice(args...; kws...)

"""
    togpudevice(x)

Move data to gpu device, backend selected by `MLDataDevices.gpu_backend!`.
"""
@inline togpudevice(args...; kws...) = toxdevice(MLDataDevices.gpu_device(), args...; kws...)

tocpudevice(args...; cache = IdDict()) = toxdevice(CPUDevice(), args...; cache)
tocudadevice(args...; cache = IdDict()) = toxdevice(CUDADevice(), args...; cache)
toamdgpudevice(args...; cache = IdDict()) = toxdevice(AMDGPUDevice(), args...; cache)
tometaldevice(args...; cache = IdDict()) = toxdevice(MetalDevice(), args...; cache)

toxdevice(adaptor::AbstractDevice, x; cache = IdDict()) = _toxdevice(adaptor, x, cache)
function toxdevice(adaptor::AbstractDevice, x, xs...; cache = IdDict())
    return (toxdevice(adaptor, x; cache), map(xi->toxdevice(adaptor, xi; cache), xs)...)
end
toxdevice(adaptor::AbstractDevice, x::Tuple; cache = IdDict()) = toxdevice(adaptor, x...; cache)
toxdevice(adaptor::AbstractDevice, x::Tuple{Any}; cache = IdDict()) = (toxdevice(adaptor, x...; cache),)
toxdevice(adaptor::AbstractDevice, x::NamedTuple{name}; cache = IdDict()) where name =
    NamedTuple{name}(toxdevice(adaptor, values(x); cache))

struct AdaptorCache{A, C} <: AbstractDict{Any, Any}
    adaptor::A
    cache::C
end
Base.haskey(cache::AdaptorCache, x) = haskey(cache.cache, x)
Base.iterate(cache::AdaptorCache, state...) = iterate(cache.cache, state...)
Base.setindex!(cache::AdaptorCache, value, key) = setindex!(cache.cache, value, key)
function __cacheget_generator__(world, source, self, cache, x)
    adaptor = cache.parameters[1]
    RT = Core.Compiler.return_type(Adapt.adapt, Tuple{adaptor, x}, world)
    body = Expr(:call, GlobalRef(Base, :getindex), Expr(:., :cache, QuoteNode(:cache)), :x)
    body = Expr(:(::), body, RT)
    expr = Expr(:lambda, [Symbol("#self#"), :cache, :x],
                Expr(Symbol("scope-block"), Expr(:block, Expr(:return, body))))
    ci = ccall(:jl_expand, Any, (Any, Any), expr, @__MODULE__)
    ci.inlineable = true
    return ci
end
@eval function Base.getindex(cache::AdaptorCache, x)
    $(Expr(:meta, :generated, __cacheget_generator__))
    $(Expr(:meta, :generated_only))
end
# https://github.com/FluxML/Functors.jl/blob/cfc6a608e309c64e4da0f44cd937cb9efa4fd6c7/src/walks.jl#L190
# CachedWalk + AdaptorCache: CachedWalk only take cache::IdDict, so we made our own
struct AdaptorWalk{W<:Functors.AbstractWalk, C<:AdaptorCache} <: Functors.AbstractWalk
    walk::W
    cache::C
end
function (walk::AdaptorWalk)(recurse, x, ys...)
    should_cache = Functors.usecache(walk.cache, x)
    if should_cache && haskey(walk.cache, x)
        return walk.cache[x]
    else
        ret = walk.walk(recurse, x, ys...)
        if should_cache
            walk.cache[x] = ret
        end
        return ret
    end
end

# https://github.com/LuxDL/Lux.jl/blob/main/lib/MLDataDevices/src/public.jl
# `(::AbstractDevice)(x)` is `Functors.fmap(Base.Fix1(Adapt.adapt, dev), x; exclude = isleaf)`;
#  we reimplement it with a type-stable cache so that shared arrays are only moved once.
@inline function __toxdevice(adaptor, cache, x, exclude, warnf)
    !isnothing(warnf) && warnf()
    walk = Functors.ExcludeWalk(Functors.DefaultWalk(), Base.Fix1(Adapt.adapt, adaptor), exclude)
    walk = AdaptorWalk(walk, AdaptorCache(adaptor, cache))
    return Functors.execute(walk, x)
end

_toxdevice(adaptor::AbstractDevice, x, cache) = __toxdevice(adaptor, cache, x, MLDataDevices.isleaf, nothing)
