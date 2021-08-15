/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#pragma once

#include <nori/object.h>
#include <nori/block.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Abstract integrator (i.e. a rendering technique)
 *
 * In Nori, the different rendering techniques are collectively referred to as 
 * integrators, since they perform integration over a high-dimensional
 * space. Each integrator represents a specific approach for solving
 * the light transport equation---usually favored in certain scenarios, but
 * at the same time affected by its own set of intrinsic limitations.
 */
class Integrator : public NoriObject {
public:
    /// Release all memory
    virtual ~Integrator() { }

    /// Perform an (optional) preprocess step
    virtual void preprocess(const Scene *scene) { }

    //每根光线追踪前的预处理,看看能不能用，害怕并行处理没有锁
    virtual void preray(const Scene *scene, Point2f bitmap_pos) { }

    //后处理步骤
    virtual void postprocess(const Scene *scene, ImageBlock &block) { }

    /**
     * \brief Sample the incident radiance along a ray
     *
     * \param scene
     *    A pointer to the underlying scene
     * \param sampler
     *    A pointer to a sample generator
     * \param ray
     *    The ray in question
     * \return
     *    A (usually) unbiased estimate of the radiance in this direction
     */
    virtual Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Point2f pixel) = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
     * provided by this instance
     * */
    EClassType getClassType() const { return EIntegrator; }
};

NORI_NAMESPACE_END
