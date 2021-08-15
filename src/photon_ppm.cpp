#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>
#include <nori/photon.h>
#include <nori/timer.h>

NORI_NAMESPACE_BEGIN

class photon_ppm : public Integrator
{
public:
    /// Photon map data structure
    typedef PointKDTree<Photon> PhotonMap;

    photon_ppm(const PropertyList &props)
    {
        photonCount = props.getInteger("photonCount", 10000);
        iteration = props.getInteger("iteration", 1 /* Default: automatic */);
        photonRadius = props.getFloat("photonRadius", 0.1f);
        alpha = props.getFloat("alpha", 0.7f);
        photonTotal = 0;
    }

    virtual void preprocess(const Scene *scene) override
    {
        m_photonMap = std::unique_ptr<PhotonMap>(new PhotonMap());
        m_photonMap->reserve(photonCount); //存每次pass的光子
        viewPointMap.reserve(4000000);     //8spp大概330w个观察点,2spp大概84w个
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Point2f bitmap_pos) override
    {

        Intersection its;
        if (scene->rayIntersect(ray, its))
        {
            Ray3f Ray(ray);
            Intersection x(its);
            Color3f wait_albedo(1.f);
            uint32_t depth = 0;
            const uint32_t LEAST_DEPTH = 5;

            while (true)
            {
                BSDFQueryRecord bRec(x.shFrame.toLocal(-Ray.d));
                Color3f albedo = x.mesh->getBSDF()->sample(bRec, sampler->next2D());
                if (albedo.maxCoeff() == 0.f)
                    break;

                if (x.mesh->isEmitter())
                {
                    EmitterQueryRecord eRec(Ray.o, x.p, x.shFrame.n);
                    Color3f radiance = x.mesh->getEmitter()->eval(eRec);
                    ViewQueryRecord vRec(x, bitmap_pos, -Ray.d, radiance, photonRadius, wait_albedo, 0);

                    viewPointMap.push_back(vRec);
                    break;
                }
                if (x.mesh->getBSDF()->isDiffuse())
                {
                    ViewQueryRecord vRec(x, bitmap_pos, -Ray.d, Color3f(0.f), photonRadius, wait_albedo, 0);
                    //观察点检索图
                    viewPointMap.push_back(vRec);

                    break;
                }

                wait_albedo *= albedo;
                Ray3f ro(x.p, x.shFrame.toWorld(bRec.wo));
                Intersection next_x;
                if (!scene->rayIntersect(ro, next_x))
                    break;
                Ray.o = ro.o;
                Ray.d = ro.d;
                x = next_x;

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
        return 0.f;
    }

    virtual void postprocess(const Scene *scene, ImageBlock &block) override
    {
        std::cout
            << "\nviewPoint nums: " << viewPointMap.size()
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
            
            //遍历viewPoint
            for (auto &vp : viewPointMap)
            {
                if (vp.its.mesh->isEmitter())
                {
                    block.put(vp.bitmap_pos, vp.power * vp.albedo);
                }
                std::vector<uint32_t> local_photon;
                m_photonMap->search(vp.its.p, vp.radius, local_photon);
                if (local_photon.size() == 0)
                    continue;
                float rate = (float)(vp.p_nums + alpha * local_photon.size()) / (vp.p_nums + local_photon.size());
                //update the radius
                vp.radius *= sqrt(rate);
                //update the power
                Color3f nPower(0.f);
                for (auto idx : local_photon)
                {
                    Photon &photon = (*m_photonMap)[idx];
                    BSDFQueryRecord bRec(vp.its.shFrame.toLocal(photon.getDirection()), vp.its.shFrame.toLocal(vp.dir), ESolidAngle);
                    Color3f fr = vp.its.mesh->getBSDF()->eval(bRec);
                    nPower += fr * photon.getPower();
                }
                vp.power = (vp.power + nPower) * rate * vp.albedo;
                vp.p_nums += local_photon.size() * alpha;
                //实时效果
                
                Point2f pos = vp.bitmap_pos;
                Color3f power = vp.power / (M_PI * vp.radius * vp.radius * photonTotal);

                block.put(pos, power);
            }
            cout << "(this pass took " << timer.elapsedString() << ")" << endl;
        }
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "PhotonMapper[\n"
            "]");
    }

private:
    uint32_t photonCount;                      //单pass有效光子数
    uint32_t photonTotal;                       //全部发射光子数
    uint32_t iteration;                        //pass次数
    float photonRadius;                        //初始半径
    float alpha;                               //衰减系数
    std::vector<ViewQueryRecord> viewPointMap; //观察点map
    std::unique_ptr<PhotonMap> m_photonMap;    //光子map
};

NORI_REGISTER_CLASS(photon_ppm, "photon_ppm");
NORI_NAMESPACE_END