package simple_estimator

type SimpleEstimator interface {
	Estimate(inputs [3]int) float64
	UpdateWithEstimation(inputs [3]int, estimation float64)
	LearningRate() float64
	GetWeights() [][][]float64
	SetWeights(weights [][][]float64)
}

type SimpleEstimatorImpl struct {
	state [][][]float64
	// Internal state is a 3 dimensional array
	// like this:
	// [
	//   [ [x1, y1, z1], [x2, y2, z2], ... ],
	//   [ [x1, y1, z1], [x2, y2, z2], ... ],
	//   ...
	// ]
	// By default all values are initialized to 1.0

	// If variables are: KEYWORD, COUNTRY, LANGUAGE
	// then the first dimension is KEYWORD, the second is COUNTRY
	// and the third is LANGUAGE.
	// The first dimension is the most important, the second is less
	// important and the third is the least important.
	// Each country corresponds always to the same index in the second
	// dimension, and each language corresponds always to the same index
	// in the third dimension.
	learningRate float64 // by default 0.1
	defaultValue float64 // by default 1.0
}

func (s *SimpleEstimatorImpl) Estimate(inputs [3]int) float64 {

	// Find the first dimension entry. If not found, add it.
	s.ensureFirstDimensionEntry(inputs[0])
	// Get value for second, and third dimension
	s.ensureSecondDimensionEntry(inputs[0], inputs[1])
	s.ensureThirdDimensionEntry(inputs[0], inputs[1], inputs[2])

	// Return the value
	return s.state[inputs[0]][inputs[1]][inputs[2]]
}

func (s *SimpleEstimatorImpl) UpdateWithEstimation(index [3]int, estimation float64) {
	currentValue := s.state[index[0]][index[1]][index[2]]
	s.state[index[0]][index[1]][index[2]] = (1.0-s.learningRate)*currentValue + s.learningRate*estimation
}

func (s *SimpleEstimatorImpl) LearningRate() float64 {
	return s.learningRate
}

func (s *SimpleEstimatorImpl) GetWeights() [][][]float64 {
	var stateCopy = make([][][]float64, len(s.state))
	for i := 0; i < len(s.state); i++ {
		stateCopy[i] = make([][]float64, len(s.state[i]))
		for j := 0; j < len(s.state[i]); j++ {
			stateCopy[i][j] = make([]float64, len(s.state[i][j]))
			for k := 0; k < len(s.state[i][j]); k++ {
				stateCopy[i][j][k] = s.state[i][j][k]
			}
		}
	}
	return stateCopy
}

func (s *SimpleEstimatorImpl) SetWeights(weights [][][]float64) {
	s.state = make([][][]float64, len(weights))
	for i := 0; i < len(weights); i++ {
		s.state[i] = make([][]float64, len(weights[i]))
		for j := 0; j < len(weights[i]); j++ {
			s.state[i][j] = make([]float64, len(weights[i][j]))
			for k := 0; k < len(weights[i][j]); k++ {
				s.state[i][j][k] = weights[i][j][k]
			}
		}
	}
}

func (s *SimpleEstimatorImpl) ensureFirstDimensionEntry(positionAtFirstDimension int) {
	// Add new entries
	for len(s.state) <= positionAtFirstDimension {
		s.state = append(s.state, make([][]float64, 0))
	}
}

func (s *SimpleEstimatorImpl) ensureSecondDimensionEntry(positionAtFirstDimension int, positionAtSecondDimension int) {
	// Add new entries
	for len(s.state[positionAtFirstDimension]) <= positionAtSecondDimension {
		s.state[positionAtFirstDimension] = append(s.state[positionAtFirstDimension], make([]float64, 0))
	}
}

func (s *SimpleEstimatorImpl) ensureThirdDimensionEntry(positionAtFirstDimension int, positionAtSecondDimension int, positionAtThirdDimension int) {
	// Add new entries
	for len(s.state[positionAtFirstDimension][positionAtSecondDimension]) <= positionAtThirdDimension {
		s.state[positionAtFirstDimension][positionAtSecondDimension] = append(s.state[positionAtFirstDimension][positionAtSecondDimension], s.defaultValue)
	}
}

// NewSimpleEstimator creates a new SimpleEstimator
// with the given default value and learning rate.
// The default value is used to initialize the internal
// state. The learning rate is used to update the internal
// state.
// Recommended values are:
// - defaultValue: 1.0
// - learningRate: [0.01, 0.49]
func NewSimpleEstimator(defaultValue float64, learningRate float64) SimpleEstimator {
	return &SimpleEstimatorImpl{
		state:        make([][][]float64, 0),
		learningRate: learningRate,
		defaultValue: defaultValue,
	}
}
