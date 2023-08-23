package tests

import (
	simpleestimator "github.com/fredyk/go-simple-estimator/simple-estimator"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
}

func TestEstimate(t *testing.T) {
	t.Parallel()

	estimator := simpleestimator.NewSimpleEstimator(10.0, 0.4)

	position := [3]int{1, 2, 3}
	estimatedValue := estimator.Estimate(position)
	assert.Equal(t, 10.0, estimatedValue)

}

func TestUpdateWithEstimation(t *testing.T) {

	t.Parallel()

	estimator := simpleestimator.NewSimpleEstimator(10.0, 0.4)

	PositionForVariableOne := 2
	PositionForVariableTwo := 2
	PositionForVariableThree := 2
	position := [3]int{PositionForVariableOne, PositionForVariableTwo, PositionForVariableThree}
	estimatedValue := estimator.Estimate(position)
	assert.Equal(t, 10.0, estimatedValue)

	realValue := 13.2
	estimator.UpdateWithEstimation([3]int{PositionForVariableOne, PositionForVariableTwo, PositionForVariableThree}, realValue)

	newEstimatedValue := estimator.Estimate(position)
	// New value should be updated with a learning rate of 0.4
	expectedNewEstimatedValue := (1.0-0.4)*estimatedValue + 0.4*realValue
	assert.Equal(t, expectedNewEstimatedValue, newEstimatedValue)

	// Again
	estimator.UpdateWithEstimation([3]int{PositionForVariableOne, PositionForVariableTwo, PositionForVariableThree}, realValue)
	newEstimatedValue = estimator.Estimate(position)
	expectedNewEstimatedValue = (1.0-0.4)*expectedNewEstimatedValue + 0.4*realValue
	assert.Equal(t, expectedNewEstimatedValue, newEstimatedValue)

	// Again
	estimator.UpdateWithEstimation([3]int{PositionForVariableOne, PositionForVariableTwo, PositionForVariableThree}, realValue)
	newEstimatedValue = estimator.Estimate(position)
	expectedNewEstimatedValue = (1.0-0.4)*expectedNewEstimatedValue + 0.4*realValue
	assert.Equal(t, expectedNewEstimatedValue, newEstimatedValue)

	// Again
	estimator.UpdateWithEstimation([3]int{PositionForVariableOne, PositionForVariableTwo, PositionForVariableThree}, realValue)
	newEstimatedValue = estimator.Estimate(position)
	expectedNewEstimatedValue = (1.0-0.4)*expectedNewEstimatedValue + 0.4*realValue
	assert.Equal(t, expectedNewEstimatedValue, newEstimatedValue)

	// save the weights

	weights := estimator.GetWeights()
	assert.Equal(t, 12.78528, weights[PositionForVariableOne][PositionForVariableTwo][PositionForVariableThree])

	// load the weights
	estimator2 := simpleestimator.NewSimpleEstimator(10.0, 0.4)
	estimator2.SetWeights(weights)

	// check the weights
	estimatedValue = estimator2.Estimate(position)
	assert.Equal(t, expectedNewEstimatedValue, estimatedValue)

}
