#include "sitkImageRegistrationMethod.h"

#include "itkNormalVariateGenerator.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkGradientDescentOptimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkLBFGSBOptimizerv4.h"
#include "itkQuasiNewtonOptimizerv4.h"
#include "itkAmoebaOptimizerv4.h"
#include "itkExhaustiveOptimizerv4.h"
#include "itkOnePlusOneEvolutionaryOptimizerv4.h"
#include "itkPowellOptimizerv4.h"


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

  template< typename TValue, typename TType>
  itk::Array<TValue> sitkSTLVectorToITKArray( const std::vector< TType > & in )
  {
    itk::Array<TValue> out(in.size());
    std::copy(in.begin(), in.end(), out.begin());
    return out;
  }

  itk::ObjectToObjectOptimizerBaseTemplate<double>*
  ImageRegistrationMethod::CreateOptimizer( unsigned int numberOfTransformParameters )
  {
    typedef double InternalComputationValueType;

    if ( m_OptimizerType == ConjugateGradientLineSearch )
      {
      typedef itk::ConjugateGradientLineSearchOptimizerv4Template<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();
      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations );
      optimizer->SetMinimumConvergenceValue( this->m_OptimizerConvergenceMinimumValue );
      optimizer->SetConvergenceWindowSize( this->m_OptimizerConvergenceWindowSize );
      optimizer->SetLowerLimit( this->m_OptimizerLineSearchLowerLimit);
      optimizer->SetUpperLimit( this->m_OptimizerLineSearchUpperLimit);
      optimizer->SetEpsilon( this->m_OptimizerLineSearchEpsilon);
      optimizer->SetMaximumLineSearchIterations( this->m_OptimizerLineSearchMaximumIterations);

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetCurrentMetricValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == GradientDescent )
      {
      typedef itk::GradientDescentOptimizerv4Template<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();
      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations );
      optimizer->SetMinimumConvergenceValue( this->m_OptimizerConvergenceMinimumValue );
      optimizer->SetConvergenceWindowSize( this->m_OptimizerConvergenceWindowSize );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetCurrentMetricValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == GradientDescentLineSearch )
      {
      typedef itk::GradientDescentLineSearchOptimizerv4Template<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();
      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations );
      optimizer->SetMinimumConvergenceValue( this->m_OptimizerConvergenceMinimumValue );
      optimizer->SetConvergenceWindowSize( this->m_OptimizerConvergenceWindowSize );
      optimizer->SetLowerLimit( this->m_OptimizerLineSearchLowerLimit);
      optimizer->SetUpperLimit( this->m_OptimizerLineSearchUpperLimit);
      optimizer->SetEpsilon( this->m_OptimizerLineSearchEpsilon);
      optimizer->SetMaximumLineSearchIterations( this->m_OptimizerLineSearchMaximumIterations);

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
      optimizer->SetGradientMagnitudeTolerance( this->m_OptimizerGradientMagnitudeTolerance );
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
      optimizer->SetMaximumNumberOfIterations( this->m_OptimizerNumberOfIterations );
      optimizer->SetMaximumNumberOfCorrections( this->m_OptimizerMaximumNumberOfCorrections  );
      optimizer->SetMaximumNumberOfFunctionEvaluations( this->m_OptimizerMaximumNumberOfFunctionEvaluations );
      optimizer->SetCostFunctionConvergenceFactor( this->m_OptimizerCostFunctionConvergenceFactor );

      #define NOBOUND     0 // 00
      #define LOWERBOUND  1 // 01
      #define UPPERBOUND  2 // 10
      #define BOTHBOUND   3 // 11

      unsigned char flag = NOBOUND;
      const unsigned int sitkToITK[] = {0,1,3,2};
      if ( this->m_OptimizerLowerBound != std::numeric_limits<double>::min() )
        {
        flag |= LOWERBOUND;
        }
      if ( this->m_OptimizerUpperBound != std::numeric_limits<double>::max() )
        {
        flag |= UPPERBOUND;
        }

      _OptimizerType::BoundSelectionType boundSelection( numberOfTransformParameters );
      _OptimizerType::BoundValueType lowerBound( numberOfTransformParameters );
      _OptimizerType::BoundValueType upperBound( numberOfTransformParameters );

      boundSelection.Fill( sitkToITK[flag] );
      lowerBound.Fill( this->m_OptimizerUpperBound );
      upperBound.Fill( this->m_OptimizerUpperBound );

      optimizer->SetBoundSelection( boundSelection );
      optimizer->SetLowerBound( lowerBound  );
      optimizer->SetUpperBound( upperBound  );
      optimizer->Register();

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == QuasiNewton )
      {
      typedef itk::QuasiNewtonOptimizerv4Template<InternalComputationValueType> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();
      optimizer->SetLearningRate( this->m_OptimizerLearningRate );
      optimizer->SetNumberOfIterations( this->m_OptimizerNumberOfIterations );
      optimizer->SetMinimumConvergenceValue( this->m_OptimizerConvergenceMinimumValue );
      optimizer->SetConvergenceWindowSize( this->m_OptimizerConvergenceWindowSize );
      optimizer->SetMaximumIterationsWithoutProgress( this->m_OptimizerMaximumIterationsWithoutProgress );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetCurrentMetricValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);
      }
    else if( m_OptimizerType == Amoeba )
      {
      typedef itk::AmoebaOptimizerv4 _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();

      optimizer->SetMaximumNumberOfIterations( this->m_OptimizerNumberOfIterations  );
      optimizer->SetParametersConvergenceTolerance(this->m_OptimizerParametersConvergenceTolerance);
      optimizer->SetFunctionConvergenceTolerance(this->m_OptimizerFunctionConvergenceTolerance);

      _OptimizerType::ParametersType simplexDelta( numberOfTransformParameters );
      simplexDelta.Fill( this->m_OptimizerSimplexDelta );
      optimizer->SetInitialSimplexDelta( simplexDelta );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == Exhaustive )
      {
      typedef itk::ExhaustiveOptimizerv4<double> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();

      optimizer->SetStepLength( this->m_OptimizerStepLength );
      optimizer->SetNumberOfSteps( sitkSTLVectorToITKArray<_OptimizerType::StepsType::ValueType>(this->m_OptimizerNumberOfSteps));

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetCurrentValue,optimizer);
      //this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer); // It will come to ITK
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == OnePlusOneEvolutionary )
      {
      typedef itk::OnePlusOneEvolutionaryOptimizerv4<double> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();

      optimizer->SetMaximumIteration( this->m_OptimizerNumberOfIterations  );

      typedef itk::Statistics::NormalVariateGenerator  GeneratorType;
      GeneratorType::Pointer generator = GeneratorType::New();
      generator->Initialize(12345);
      optimizer->SetNormalVariateGenerator( generator );

      optimizer->Initialize( this->m_OptimizerInitialRadius );
      optimizer->SetEpsilon( this->m_OptimizerEpsilon );
      optimizer->SetGrowthFactor( this->m_OptimizerGrowthFactor );
      optimizer->SetShrinkFactor( this->m_OptimizerShrinkFactor );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);

      optimizer->Register();
      return optimizer.GetPointer();
      }
    else if ( m_OptimizerType == Powell )
      {
      typedef itk::PowellOptimizerv4<double> _OptimizerType;
      _OptimizerType::Pointer      optimizer     = _OptimizerType::New();

      optimizer->SetStepLength( this->m_OptimizerStepLength );
      optimizer->SetStepTolerance( this->m_StepTolerance );
      optimizer->SetValueTolerance( this->m_ValueTolerance );
      optimizer->SetMaximumIteration( this->m_OptimizerNumberOfIterations );

      this->m_pfGetMetricValue = nsstd::bind(&_OptimizerType::GetValue,optimizer);
      this->m_pfGetOptimizerIteration = nsstd::bind(&_OptimizerType::GetCurrentIteration,optimizer);
      this->m_pfGetOptimizerPosition = nsstd::bind(&PositionOptimizerCustomCast::CustomCast,optimizer);
      }
    else
      {
      sitkExceptionMacro("LogicError: Unexpected case!");
      }
  }

}
}
