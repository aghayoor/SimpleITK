#ifndef __sitkImageRegistrationMethod_h
#define __sitkImageRegistrationMethod_h

#include "sitkRegistration.h"

#include "sitkDetail.h"
#include "sitkImage.h"
#include "sitkPixelIDTokens.h"
#include "sitkMemberFunctionFactory.h"
#include "sitkProcessObject.h"

#include "sitkInterpolator.h"
#include "sitkTransform.h"

#ifndef SWIGPYTHON
#define LAMBDA lambda
#else
#define LAMBDA _lambda
#endif



namespace itk
{

#ifndef SWIG
class SingleValuedNonLinearOptimizer;
template<class T, class U> class ImageToImageMetric;
class Command;
class EventObject;
#endif



namespace simple
{

  class SITKRegistration_EXPORT ImageRegistrationMethod
    : public ProcessObject
  {
  public:

    typedef ImageRegistrationMethod Self;
    typedef ProcessObject Superclass;

    ImageRegistrationMethod();
    virtual ~ImageRegistrationMethod();

    std::string GetName() const { return std::string("ImageRegistrationMethod"); }
    std::string ToString() const;


    InterpolatorEnum GetInterpolator()
      { return this->m_Interpolator; }
    Self& SetInterpolator ( InterpolatorEnum Interpolator )
      { this->m_Interpolator = Interpolator; return *this; }

    Self& SetTransform ( const Transform &Transform )
      { this->m_Transform = Transform; return *this; }
    Transform GetTransform()
      { return this->m_Transform; }

    Self& SetMetricAsMeanSquares( uint64_t numberOfSpatialSamples = 0 );
    Self& SetMetricAsMutualInformation( double fixedImageStandardDeviation=0.4,
                                        double movingImageStandardDeviation=0.4,
                                        uint64_t numberOfSpatialSamples = 50 );
    Self& SetMetricAsMattesMutualInformation( unsigned int numberOfHistogramBins = 50,
                                              bool useExplicitPDFDerivatives=true,
                                              uint64_t numberOfSpatialSamples = 0 );


    Self& SetOptimizerAsRegularStepGradientDescent( double maxStep,
                                                    double minStep,
                                                    unsigned int numberOfIterations,
                                                    double relaxationFactor=0.5);
    Self& SetOptimizerAsGradientDescent( double learningRate,
                                         unsigned int numberOfIterations );
    Self& SetOptimizerScales( const std::vector<double> &scales);


    Self& SetFixedImageRegion( const std::vector<unsigned int> &size, const std::vector<unsigned int> &index);

    Transform Execute ( const Image &fixed, const Image & moving );

     /**
      * Active measurements which can be obtained during call backs.
      *
      * This is a measurement. Its value is updated in the Execute
      * methods, so the value will only be valid after an execution.
      */
    unsigned int GetOptimizerIteration() const;
    std::vector<double> GetOptimizerPosition() const;
    double GetMetricValue() const;

    /** Measurement updated at the end of execution.
      */
    const std::string &GetOptimizerStopConditionDescription() const;

  protected:

    template<class TImage>
    Transform ExecuteInternal ( const Image &fixed, const Image &moving );

    itk::SingleValuedNonLinearOptimizer* CreateOptimizer( );

    template <class TImageType>
      itk::ImageToImageMetric<TImageType,TImageType>* CreateMetric( );

    virtual void PreUpdate( itk::ProcessObject *p );
    virtual void OnActiveProcessDelete( ) throw();
    virtual unsigned long AddITKObserver(const itk::EventObject &, itk::Command *);
    virtual void RemoveITKObserver( EventCommand &e );

  private:

    nsstd::function<unsigned int()> m_pfGetOptimizerIteration;
    nsstd::function<std::vector<double>()> m_pfGetOptimizerPosition;
    nsstd::function<double()> m_pfGetMetricValue;

    typedef Transform (ImageRegistrationMethod::*MemberFunctionType)( const Image &fixed, const Image &moving );
    friend struct detail::MemberFunctionAddressor<MemberFunctionType>;
    std::auto_ptr<detail::MemberFunctionFactory<MemberFunctionType> > m_MemberFactory;

    InterpolatorEnum  m_Interpolator;
    Transform  m_Transform;

    // optimizer
    enum OptimizerType { RegularStepGradientDescent,
                         GradientDescent
    };
    OptimizerType m_OptimizerType;
    double m_OptimizerMaximumStepLength;
    double m_OptimizerMinimumStepLength;
    unsigned int m_OptimizerNumberOfIterations;
    double m_OptimizerRelaxationFactor;
    bool m_OptimizerMinimize;
    std::vector<double> m_OptimizerScales;
    double m_OptimizerLearningRate;

    // metric
    enum MetricType { MeanSquares,
                      MattesMutualInformation,
                      MutualInformation
    };

    MetricType m_MetricType;
    uint64_t  m_MetricNumberOfSpatialSamples;
    double m_MetricFixedImageStandardDeviation;
    double m_MetricMovingImageStandardDeviation;
    unsigned int m_MetricNumberOfHistogramBins;
    bool m_MetricUseExplicitPDFDerivatives;

    std::vector<unsigned int> m_FixedImageRegionSize;
    std::vector<unsigned int> m_FixedImageRegionIndex;


    std::string m_StopConditionDescription;
    double m_MetricValue;
    unsigned int m_Iteration;

    itk::SingleValuedNonLinearOptimizer *m_ActiveOptimizer;
  };

}
}

#endif // __sitkImageRegistrationMethod_h
