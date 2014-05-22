#include "sitkImageRegistrationMethod.h"

#include "itkGradientDescentOptimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkLBFGSBOptimizerv4.h"


namespace {

struct PositionOptimizerCustomCast
{
  template <typename T>
  static std::vector<double> Helper(const T &value)
    { return std::vector<double>(value.begin(),value.end()); }

  static std::vector<double> CustomCast(const itk::ObjectToObjectOptimizerBaseTemplate<double> *opt)
    {
      return Helper(opt->GetCurrentPosition());
    }
};

}

namespace itk
{
namespace simple
{



  itk::ObjectToObjectOptimizerBaseTemplate<double>*
  ImageRegistrationMethod::CreateOptimizer( )
  {
    typedef double InternalComputationValueType;

    if ( m_OptimizerType == GradientDescent )
      {
      typedef itk::GradientDescentOptimizerv4Template<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();
      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations  );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetCurrentMetricValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == RegularStepGradientDescent )
      {
      typedef itk::RegularStepGradientDescentOptimizerv4<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer =  _OptimizerType::New();

      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetMinimumStepLength( this->m_OptimizerMinimumStepLength );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations  );
      optimizer->SetRelaxationFactor( this->m_OptimizerRelaxationFactor );
      optimizer->Register();

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == LBFGSB )
      {
      typedef itk::LBFGSBOptimizerv4 _OptimizerType;
      _OptimizerType::Pointer      optimizer =  _OptimizerType::New();

      optimizer->SetGradientConvergenceTolerance( this->m_OptimizerGradientConvergenceTolerance );
      optimizer->SetMaximumNumberOfIterations( this->m_OptimizerMaximumNumberOfIterations );
      optimizer->SetMaximumNumberOfCorrections( this->m_OptimizerMaximumNumberOfCorrections  );
      optimizer->SetMaximumNumberOfFunctionEvaluations( this->m_OptimizerMaximumNumberOfFunctionEvaluations );
      optimizer->SetCostFunctionConvergenceFactor( this->m_OptimizerCostFunctionConvergenceFactor );

      _OptimizerType::BoundSelectionType boundSelection(/* numberOfTransformParameters */);
      _OptimizerType::BoundValueType upperBound(/* numberOfTransformParameters */);
      _OptimizerType::BoundValueType lowerBound(/* numberOfTransformParameters */);
      if ( this->m_OptimizerUpperBound != std::numeric_limits<double>::max() &&
           this->m_OptimizerLowerBound != std::numeric_limits<double>::min() )
        {
        upperBound.Fill( this->m_OptimizerUpperBound );
        lowerBound.Fill( this->m_OptimizerLowerBound );
        boundSelection.Fill( 2 ); // Optimizer has both Lower and Upper bounds
        }
      else if ( this->m_OptimizerUpperBound != std::numeric_limits<double>::max() )
        {
        upperBound.Fill( this->m_OptimizerUpperBound );
        lowerBound.Fill( 0 );
        boundSelection.Fill( 3 ); // Optimizer has only Upper bounds
        }
      else if ( this->m_OptimizerLowerBound != std::numeric_limits<double>::min() )
        {
        upperBound.Fill( 0 );
        lowerBound.Fill( this->m_OptimizerLowerBound );
        boundSelection.Fill( 1 ); // Optimizer has only Lower bounds
        }
      else
        {
        upperBound.Fill( 0 );
        lowerBound.Fill( 0 );
        boundSelection.Fill( 0 ); // Optimizer is unbounded
        }
      optimizer->SetBoundSelection( boundSelect );
      optimizer->SetUpperBound( upperBound );
      optimizer->SetLowerBound( lowerBound );

      optimizer->Register();

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      return optimizer.GetPointer();
      }
    else
      {
      sitkExceptionMacro("LogicError: Unexpected case!");
      }
  }

}
}
