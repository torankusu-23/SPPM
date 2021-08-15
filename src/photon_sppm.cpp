#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>
#include <nori/photon.h>
#include <nori/timer.h>
#include <nori/camera.h>

NORI_NAMESPACE_BEGIN

class photon_sppm : public Integrator
{
public:
    /// Photon map data structure
    typedef PointKDTree<Photon> PhotonMap;

    photon_sppm(const PropertyList &props)
    {
        photonCount = props.getInteger("photonCount", 10000);
        iteration = props.getInteger("iteration", 1 /* Default: automatic */);
        sharedRadius = props.getFloat("photonRadius", 0.1f);
        alpha = props.getFloat("alpha", 0.7f);
        photonTotal = 0;
    }

    virtual void preprocess(const Scene *scene) override
    {
        m_photonMap = std::unique_ptr<PhotonMap>(new PhotonMap());
        m_photonMap->reserve(photonCount); //存每次pass的光子
        PixelMap.reserve(480001);          //1spp,48w
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Point2f pixel) override
    {
        PixelQueryRecord cur(pixel, sharedRadius, Color3f(0.f), 0);
        PixelMap.push_back(cur);

        return 0.f;
    }

    virtual void postprocess(const Scene *scene, ImageBlock &block) override
    {
        std::cout
            << "\npixel nums: " << PixelMap.size()
            << "\niteration nums: " << iteration
            << "\nphoton nums per pass: " << photonCount
            << std::endl;
        
        Sampler *sampler = static_cast<Sampler *>(
            NoriObjectFactory::createInstance("independent", PropertyList()));
        std::vector<Mesh *> lights;
        for (auto m : scene->getMeshes())
        {
            if (m->isEmitter())
                lights.emplace_back(m);
        }
        int nLights = lights.size();
        const uint32_t LEAST_DEPTH = 5;
        const Camera *camera = scene->getCamera();

        //先发射光子pass，再分布式raytrace pass，这样可以在第二个pass中直接收集光子，而不用存储观察点
        for (uint32_t i = 0; i < iteration; ++i)
        {
            Timer timer;
            m_photonMap = std::unique_ptr<PhotonMap>(new PhotonMap());
            m_photonMap->reserve(photonCount); //存每次pass的光子

            uint32_t storedPhotons = 0;
            uint32_t photonEmitter = 0;
            //光子发射pass
            while (storedPhotons < photonCount)
            {
                Mesh *areaLight = lights[nLights * sampler->next1D()];
                //采样发射光子，这个光子的方向是特殊的，是出射方向，其他都是入射方向（为后续fr作准备）
                Photon emitPhoton = areaLight->getEmitter()->samplePhoton(sampler, areaLight, nLights);

                Ray3f Ray(emitPhoton.getPosition(), emitPhoton.getDirection());
                Intersection its;
                Color3f wait_albedo(1.f);
                uint32_t depth = 0;

                photonEmitter++;
                if (scene->rayIntersect(Ray, its))
                {
                    while (true)
                    {
                        if (its.mesh->getBSDF()->isDiffuse())
                        {
                            Photon p(its.p, -Ray.d, emitPhoton.getPower() * wait_albedo);
                            m_photonMap->push_back(p);
                            storedPhotons++;
                        }

                        BSDFQueryRecord bRec(its.shFrame.toLocal(-Ray.d));
                        Color3f albedo = its.mesh->getBSDF()->sample(bRec, sampler->next2D());
                        if (albedo.maxCoeff() == 0.f)
                            break;

                        wait_albedo *= albedo;
                        Ray3f ro(its.p, its.shFrame.toWorld(bRec.wo));
                        Intersection next_its;

                        if (!scene->rayIntersect(ro, next_its))
                        {
                            break;
                        }
                        Ray.o = ro.o;
                        Ray.d = ro.d;
                        its = next_its;

                        if (depth < LEAST_DEPTH)
                        {
                            depth++;
                        }
                        else
                        {
                            //RR
                            float q = wait_albedo.maxCoeff();
                            if (sampler->next1D() > q)
                                break;
                            wait_albedo /= q;
                        }
                    }
                }
            }
            m_photonMap->build();
            photonTotal += photonEmitter;

            //发射viewPoint pass
            for (auto &pp : PixelMap)
            {
                Ray3f ray;
                Point2f tmp = 0.f;
                Point2f sample = pp.pixel + sampler->next2D();
                camera->sampleRay(ray, sample, tmp);
                Intersection its;
                Color3f throughput(1.f);
                uint32_t depth = 0;

                if (scene->rayIntersect(ray, its))
                {
                    while (true)
                    {
                        //光源处理
                        if (its.mesh->isEmitter())
                        {
                            EmitterQueryRecord eRec(ray.o, its.p, its.shFrame.n);
                            Color3f power = its.mesh->getEmitter()->eval(eRec);
                            block.put(pp.pixel + Point2f(0.5f), power * throughput);
                            break;
                        }

                        //diffuse表面
                        if (its.mesh->getBSDF()->isDiffuse())
                        {
                            std::vector<uint32_t> local_photon;
                            m_photonMap->search(its.p, pp.radius, local_photon);
                            if (local_photon.size() == 0)
                                break;
                            float rate = (float)(pp.p_nums + alpha * local_photon.size()) / (pp.p_nums + local_photon.size());
                            pp.p_nums += local_photon.size() * alpha;
                            pp.radius = pp.radius * rate;
                            Color3f nPower(0.f);
                            for (auto idx : local_photon)
                            {
                                Photon &photon = (*m_photonMap)[idx];
                                BSDFQueryRecord bRec(its.shFrame.toLocal(-ray.d), its.shFrame.toLocal(photon.getDirection()), ESolidAngle);
                                Color3f fr = its.mesh->getBSDF()->eval(bRec);
                                nPower += fr * photon.getPower();
                            }
                            pp.flux = (pp.flux + nPower) * rate * throughput;
                            break;
                        }

                        //specular
                        BSDFQueryRecord bRec(its.shFrame.toLocal(-ray.d));
                        Color3f albedo = its.mesh->getBSDF()->sample(bRec, sampler->next2D());
                        if (albedo.maxCoeff() == 0.f)
                            break;

                        throughput *= albedo;
                        Ray3f ro(its.p, its.shFrame.toWorld(bRec.wo));
                        Intersection next_its;
                        if (!scene->rayIntersect(ro, next_its))
                            break;
                        ray.o = ro.o;
                        ray.d = ro.d;
                        its = next_its;

                        if (depth < LEAST_DEPTH)
                            depth++;
                        else
                        {
                            //RR
                            float q = throughput.maxCoeff();
                            if (sampler->next1D() > q)
                                break;
                            throughput /= q;
                        }
                    }
                }

                //同步显示
                Point2f pos = pp.pixel + Point2f(0.5f);
                Color3f power = pp.flux / (photonTotal * M_PI * pp.radius * pp.radius);
                block.put(pos, power);
            }//viewPass结束
            cout << "(the "<<i+1<<" pass took " << timer.elapsedString() << ")" << endl;
        }//iter结束
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "PhotonMapper[\n"
            "]");
    }

private:
    uint32_t photonCount;                   //单pass有效光子数
    uint32_t photonTotal;                   //全部发射光子数
    uint32_t iteration;                     //pass次数
    float sharedRadius;                     //初始半径
    float alpha;                            //衰减系数
    std::vector<PixelQueryRecord> PixelMap; //像素map
    std::unique_ptr<PhotonMap> m_photonMap; //光子map
};

NORI_REGISTER_CLASS(photon_sppm, "photon_sppm");
NORI_NAMESPACE_END