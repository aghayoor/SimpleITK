/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __sitkPimpleTransform_h
#define __sitkPimpleTransform_h

#include "sitkTransform.h"

#include "itkTransformBase.h"


#include "itkIdentityTransform.h"
#include "itkTranslationTransform.h"
#include "itkScaleTransform.h"
#include "itkScaleLogarithmicTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkSimilarity3DTransform.h"
#include "itkEuler2DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkVersorTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkAffineTransform.h"
#include "itkCompositeTransform.h"

#include "itkDisplacementFieldTransform.h"
#include "itkBSplineTransform.h"

namespace itk
{
namespace simple
{

// This is a base class of the private implementatino of the transform
// class.
//
// The interface provide virutal method and other generic methods to
// the concrete ITK transform type, there by provide encapsulation and
// a uniform interface
class PimpleTransformBase
{
public:
  virtual ~PimpleTransformBase( void ) {};

  // Get Access to the internal ITK transform class
  virtual TransformBase::Pointer GetTransformBase( void ) = 0;
  virtual TransformBase::ConstPointer GetTransformBase( void ) const = 0;

  // general methods to get information about the internal class
  virtual unsigned int GetInputDimension( void ) const = 0;
  virtual unsigned int GetOutputDimension( void ) const = 0;

  // Set the fixed parameter for the transform, converting from the
  // simpleITK std::vector to the ITK's array.
  void SetFixedParameters( const std::vector< double > &inParams )
    {
      itk::TransformBase::ParametersType p( inParams.size() );

      // todo check expected number of Fixed parameters
      std::copy( inParams.begin(), inParams.end(), p.begin() );
      this->GetTransformBase()->SetFixedParameters( p );
    }

  // Get the fixed parameters form the transform
  std::vector< double >  GetFixedParameters( void ) const
    {
      const itk::TransformBase::ParametersType &p = this->GetTransformBase()->GetFixedParameters();
      return std::vector< double >( p.begin(), p.end() );
    }


  unsigned int GetNumberOfParameters( void ) const { return this->GetTransformBase()->GetNumberOfParameters(); }


  void SetParameters( const std::vector< double > &inParams )
    {
      itk::TransformBase::ParametersType p( inParams.size() );

      // todo check expected number of Parameters
      std::copy( inParams.begin(), inParams.end(), p.begin() );
      this->GetTransformBase()->SetParameters( p );
    }
  std::vector< double > GetParameters( void ) const
    {
      const itk::TransformBase::ParametersType &p = this->GetTransformBase()->GetParameters();
      return std::vector< double >( p.begin(), p.end() );
    }


  virtual PimpleTransformBase *ShallowCopy( void ) const = 0;
  virtual PimpleTransformBase *DeepCopy( void ) const = 0;

  virtual int GetReferenceCount( ) const = 0;

  std::string ToString( void ) const
    {
      std::ostringstream out;
      this->GetTransformBase()->Print ( out );
      return out.str();
    }

  // note: the returned pointer needs to be externally managed and
  // deleted
  // Also the return pointer could be this
  virtual PimpleTransformBase* AddTransform( Transform &t ) = 0;


  virtual std::vector< double > TransformPoint( const std::vector< double > &t ) const = 0;

protected:

};

template< typename TTransformType >
class PimpleTransform
  : public PimpleTransformBase
{
public:
  typedef PimpleTransform                  Self;
  typedef TTransformType                   TransformType;
  typedef typename TransformType::Pointer  TransformPointer;

  typedef itk::CompositeTransform<double, TransformType::InputSpaceDimension> CompositeTransformType;

  static const unsigned int InputDimension = TTransformType::InputSpaceDimension;
  static const unsigned int OutputDimension = TTransformType::OutputSpaceDimension;

  template<typename TScalar>
  PimpleTransform( itk::Image<itk::Vector<TScalar, InputDimension>, InputDimension> *displacement )
    {
      this->m_Transform = TransformType::New();
      this->m_Transform->SetDisplacementField( displacement );
    }


  PimpleTransform( TransformType * p)
    {
      this->m_Transform = p;
    }

  PimpleTransform( )
    {
      this->m_Transform = TransformType::New();
    }

  PimpleTransform( Self &s )
    : m_Transform( s.m_Transform )
    {}

  PimpleTransform &operator=( const PimpleTransform &s )
    {
      m_Transform = s.m_Transform;
    }

  virtual TransformBase::Pointer GetTransformBase( void ) { return this->m_Transform.GetPointer(); }
  virtual TransformBase::ConstPointer GetTransformBase( void ) const { return this->m_Transform.GetPointer(); }

  virtual unsigned int GetInputDimension( void ) const { return InputDimension; }
  virtual unsigned int GetOutputDimension( void ) const { return OutputDimension; }


  virtual PimpleTransformBase *ShallowCopy( void ) const
    {
      return new Self( this->m_Transform.GetPointer() );
    }

  virtual PimpleTransformBase *DeepCopy( void ) const
    {
      PimpleTransformBase *copy( new Self( this->m_Transform->Clone() ) );
      return copy;
    }

  virtual int GetReferenceCount( ) const
    {
      return this->m_Transform->GetReferenceCount();
    }

  virtual PimpleTransformBase* AddTransform( Transform &t )
    {
      if ( t.GetDimension() != TransformType::InputSpaceDimension )
        {
        sitkExceptionMacro( "Transform argument has dimension " << t.GetDimension()
                            << " does not match this dimesion of " << TransformType::InputSpaceDimension );
        }

      typename CompositeTransformType::TransformType* base =
        dynamic_cast< typename CompositeTransformType::TransformType*>( t.GetITKBase() );

      return this->AddTransform( base, typename nsstd::is_same<TTransformType, CompositeTransformType>::type() );
    }

  PimpleTransformBase* AddTransform( typename CompositeTransformType::TransformType* t, nsstd::true_type isCompositeTransform )
    {
      Unused( isCompositeTransform );
      assert( t->GetInputSpaceDimension() == TransformType::InputSpaceDimension );

      m_Transform->AddTransform( t );
      m_Transform->SetAllTransformsToOptimizeOff();
      m_Transform->SetOnlyMostRecentTransformToOptimizeOn();

      return this;
    }

  PimpleTransformBase* AddTransform( typename CompositeTransformType::TransformType* t, nsstd::false_type isNotCompositeTransform )
    {
      Unused( isNotCompositeTransform );

      typename CompositeTransformType::Pointer composite = CompositeTransformType::New();
      composite->AddTransform( this->m_Transform );
      composite->AddTransform( t );
      composite->SetAllTransformsToOptimizeOff();
      composite->SetOnlyMostRecentTransformToOptimizeOn();

      return new PimpleTransform<CompositeTransformType>( composite );
    }



  virtual std::vector< double > TransformPoint( const std::vector< double > &pt ) const
    {
      if (pt.size() != this->GetInputDimension() )
        {
        sitkExceptionMacro("vector dimension mismatch");
        }

      typename TransformType::OutputPointType opt =
        this->m_Transform->TransformPoint( sitkSTLVectorToITK< typename TransformType::InputPointType >(pt));

      return sitkITKVectorToSTL<double>( opt );
    }

private:

  TransformPointer m_Transform;
};

}
}


#endif //  __sitkPimpleTransform_h
