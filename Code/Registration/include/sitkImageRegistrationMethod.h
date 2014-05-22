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
template< typename TInternalComputationValueType> class ObjectToObjectOptimizerBaseTemplate;
template<typename TFixedImage,
         typename TMovingImage,
         typename TVirtualImage,
         typename TInternalComputationValueType,
         typename TMetricTraits >
class ImageToImageMetricv4;

template<typename TFixedImageType,
         typename TMovingImageType,
         typename TVirtualImageType,
         typename TCoordRep>
class DefaultImageToImageMetricTraitsv4;

template<typename TMetric> class RegistrationParameterScalesEstimator;

template<unsigned int VDimension> class SpatialObject;

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

    Self& SetMetricAsCorrelation( );
    Self& SetMetricAsDemons( );
    Self& SetMetricAsJointHistogramMutualInformation( );
    Self& SetMetricAsMeanSquares( );
    Self& SetMetricAsMattesMutualInformation( );


    Self& SetOptimizerAsRegularStepGradientDescent( double learningRate,
                                                    double minStep,
                                                    unsigned int numberOfIterations,
                                                    double relaxationFactor=0.5);
    Self& SetOptimizerAsGradientDescent( double learningRate,
                                         unsigned int numberOfIterations );

    Self& SetOptimizerScales( const std::vector<double> &scales );
    Self& SetOptimizerScalesFromJacobian( unsigned int centralRegionRadius = 5 );
    Self& SetOptimizerScalesFromIndexShift( unsigned int centralRegionRadius = 5,
                                           double smallParameterVariation =  0.01 );
    Self& SetOptimizerScalesFromPhysicalShift( unsigned int centralRegionRadius = 5,
                                              double smallParameterVariation =  0.01 );



    Self& SetMetricFixedMask( const Image &binaryMask );

    Self& SetMetricMovingMask( const Image &binaryMask );

    Self &SetMetricSamplingPercentage(double percentage);
    Self &SetMetricSamplingPercentagePerLevel(const std::vector<double> &percentage);

    enum MetricSamplingStrategyType {
      NONE,
      REGULAR,
      RANDOM
    };

    Self &SetMetricSamplingStrategy( MetricSamplingStrategyType strategy);

    Self &SetShrinkFactorsPerLevel( const std::vector<unsigned int> &shrinkFactors );
    Self &SetSmoothingSigmasPerLevel( const std::vector<double> &smoothingSigmas );

    Self &SetSmoothingSigmasAreSpecifiedInPhysicalUnits(bool arg);
    Self &SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() { this->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(true); return *this;}
    Self &SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()  { this->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false); return *this;}


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

    itk::ObjectToObjectOptimizerBaseTemplate<double> *CreateOptimizer( );

    template<unsigned int VDimension>
      itk::SpatialObject<VDimension> *CreateSpatialObjectMask(const Image &mask);

    template <class TImageType>
      itk::ImageToImageMetricv4<TImageType,
      TImageType,
      TImageType,
      double,
      DefaultImageToImageMetricTraitsv4< TImageType, TImageType, TImageType, double >
      >* CreateMetric( );

    template <typename TMetric>
      itk::RegistrationParameterScalesEstimator< TMetric >*CreateScalesEstimator();

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
    double m_OptimizerLearningRate;
    double m_OptimizerMinimumStepLength;
    unsigned int m_OptimizerNumberOfIterations;
    double m_OptimizerRelaxationFactor;

    enum OptimizerScalesType {
      Manual,
      Jacobian,
      IndexShift,
      PhysicalShift
    };
    OptimizerScalesType m_OptimizerScalesType;
    std::vector<double> m_OptimizerScales;
    unsigned int m_OptimizerScalesCentralRegionRadius;
    double m_OptimizerScalesSmallParameterVariation;

    // metric
    enum MetricType { Correlation,
                      Demons,
                      JointHistogramMutualInformation,
                      MeanSquares,
                      MattesMutualInformation
    };
    MetricType m_MetricType;

    Image m_MetricFixedMaskImage;
    Image m_MetricMovingMaskImage;
    std::vector<unsigned int> m_MetricFixedMaskRegion;
    std::vector<unsigned int> m_MetricMovingMaskRegion;

    std::vector<double> m_MetricSamplingPercentage;
    MetricSamplingStrategyType m_MetricSamplingStrategy;

    std::vector<unsigned int> m_ShrinkFactorsPerLevel;
    std::vector<double> m_SmoothingSigmasPerLevel;
    bool m_SmoothingSigmasAreSpecifiedInPhysicalUnits;

    std::string m_StopConditionDescription;
    double m_MetricValue;
    unsigned int m_Iteration;

    itk::ObjectToObjectOptimizerBaseTemplate<double> *m_ActiveOptimizer;
  };

}
}

#endif // __sitkImageRegistrationMethod_h
