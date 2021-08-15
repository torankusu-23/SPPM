#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/sampler.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class AreaLight : public Emitter
{
public:
    AreaLight(const PropertyList &propList)
    {
        radiance = propList.getColor("radiance");
    }

    //生成采样点
    virtual Color3f sample(const Mesh *mesh, EmitterQueryRecord &eRec, Sampler *&sample) const override
    {
        SampleMeshResult point = mesh->sampleSurfaceUniform(sample);
        eRec.y = point.p;
        eRec.n = point.n;
        eRec.wi = (eRec.y - eRec.x).normalized();
        eRec.pdf = pdf(mesh, eRec); //光源面积pdf

        if (eRec.pdf > 0.0f && !std::isnan(eRec.pdf) && !std::isinf(eRec.pdf))
        {
            return eval(eRec) / eRec.pdf; //这里没有cos项，缺的都吸收在G项中
        }

        return Color3f(0.f);
    }

    //计算pdf
    virtual float pdf(const Mesh *mesh, const EmitterQueryRecord &eRec) const override
    {
        if (eRec.n.dot(-eRec.wi) > 0.f)
        {
            return mesh->getpdf().getNormalization();
        }
        return 0.f;
    }

    //返回radiance，存在一个Color3f通道中
    virtual Color3f eval(const EmitterQueryRecord &eRec) const override
    {
        return (eRec.n.dot(-eRec.wi) > 0.f) ? radiance : 0.0f; //背面采样点剔除
    }

    virtual Photon samplePhoton(Sampler *sampler, Mesh* area, uint32_t light_num) const override
    {
        //采样光源三角形
        SampleMeshResult sample = area->sampleSurfaceUniform(sampler);
        //采样cos方向
        Vector3f cosDir = Warp::squareToCosineHemisphere(sampler->next2D());
        //坐标系变换，把局部方向加到世界方向中
        Frame frame = Frame(sample.n);
        Vector3f we = frame.toWorld(cosDir);
        //能量公式，公示版本有点多，有待商榷
        Color3f power = M_PI * radiance * light_num / sample.pdf;

        return Photon(sample.p, we, power);
    }

    std::string toString() const override
    {
        return "Emitter[]";
    }

private:
    Color3f radiance;
};

NORI_REGISTER_CLASS(AreaLight, "area")
NORI_NAMESPACE_END