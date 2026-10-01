//! Continuous-time models: Neural ODE integration and liquid time-constant cells.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiquidError(pub String);
impl fmt::Display for LiquidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for LiquidError {}

/// Numerical solver used by [`NeuralOde`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSolver {
    Euler,
    Heun,
    RungeKutta4,
}

/// A neural ODE `dy/dt = f(t, y)` with a fixed-step integrator.
pub struct NeuralOde<F> {
    field: F,
    solver: OdeSolver,
}

impl<F> NeuralOde<F>
where
    F: Fn(f32, &[f32]) -> Vec<f32>,
{
    pub fn new(field: F, solver: OdeSolver) -> Self {
        Self { field, solver }
    }
    pub fn solve(
        &self,
        initial: &[f32],
        start: f32,
        end: f32,
        steps: usize,
    ) -> Result<Vec<f32>, LiquidError> {
        if initial.is_empty() || steps == 0 || !start.is_finite() || !end.is_finite() {
            return Err(LiquidError(
                "state, interval, and step count must be valid".into(),
            ));
        }
        let dt = (end - start) / steps as f32;
        let mut y = initial.to_vec();
        let mut t = start;
        for _ in 0..steps {
            y = self.step(t, &y, dt)?;
            t += dt;
        }
        Ok(y)
    }
    fn eval(&self, t: f32, y: &[f32]) -> Result<Vec<f32>, LiquidError> {
        let value = (self.field)(t, y);
        if value.len() != y.len() {
            return Err(LiquidError(
                "vector field changed the state dimension".into(),
            ));
        }
        Ok(value)
    }
    fn step(&self, t: f32, y: &[f32], h: f32) -> Result<Vec<f32>, LiquidError> {
        let k1 = self.eval(t, y)?;
        match self.solver {
            OdeSolver::Euler => Ok(add(y, &k1, h)),
            OdeSolver::Heun => {
                let k2 = self.eval(t + h, &add(y, &k1, h))?;
                Ok(combine(y, &[(&k1, h / 2.0), (&k2, h / 2.0)]))
            }
            OdeSolver::RungeKutta4 => {
                let k2 = self.eval(t + h / 2.0, &add(y, &k1, h / 2.0))?;
                let k3 = self.eval(t + h / 2.0, &add(y, &k2, h / 2.0))?;
                let k4 = self.eval(t + h, &add(y, &k3, h))?;
                Ok(combine(
                    y,
                    &[
                        (&k1, h / 6.0),
                        (&k2, h / 3.0),
                        (&k3, h / 3.0),
                        (&k4, h / 6.0),
                    ],
                ))
            }
        }
    }
}

fn add(y: &[f32], k: &[f32], scale: f32) -> Vec<f32> {
    y.iter().zip(k).map(|(a, b)| a + scale * b).collect()
}
fn combine(y: &[f32], terms: &[(&[f32], f32)]) -> Vec<f32> {
    (0..y.len())
        .map(|i| y[i] + terms.iter().map(|(v, s)| v[i] * s).sum::<f32>())
        .collect()
}

/// Liquid time-constant recurrent cell.
#[derive(Debug, Clone)]
pub struct LiquidCell {
    input_size: usize,
    hidden_size: usize,
    input_weights: Vec<f32>,
    recurrent_weights: Vec<f32>,
    bias: Vec<f32>,
    tau: Vec<f32>,
    solver: OdeSolver,
}

impl LiquidCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, LiquidError> {
        if input_size == 0 || hidden_size == 0 {
            return Err(LiquidError("layer dimensions must be positive".into()));
        }
        // Deterministic initialization makes examples and tests reproducible.
        let iw = (0..hidden_size * input_size)
            .map(|i| ((i * 17 + 3) % 23) as f32 / 46.0 - 0.25)
            .collect();
        let rw = (0..hidden_size * hidden_size)
            .map(|i| ((i * 11 + 5) % 19) as f32 / 76.0 - 0.125)
            .collect();
        Ok(Self {
            input_size,
            hidden_size,
            input_weights: iw,
            recurrent_weights: rw,
            bias: vec![0.0; hidden_size],
            tau: vec![1.0; hidden_size],
            solver: OdeSolver::RungeKutta4,
        })
    }
    pub fn with_solver(mut self, solver: OdeSolver) -> Self {
        self.solver = solver;
        self
    }
    /// Constructs a cell from row-major parameters. Time constants must be
    /// finite and strictly positive.
    pub fn from_parameters(
        input_size: usize,
        hidden_size: usize,
        input_weights: Vec<f32>,
        recurrent_weights: Vec<f32>,
        bias: Vec<f32>,
        tau: Vec<f32>,
        solver: OdeSolver,
    ) -> Result<Self, LiquidError> {
        if input_size == 0
            || hidden_size == 0
            || input_weights.len() != input_size * hidden_size
            || recurrent_weights.len() != hidden_size * hidden_size
            || bias.len() != hidden_size
            || tau.len() != hidden_size
            || tau.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(LiquidError("invalid liquid-cell parameters".into()));
        }
        Ok(Self {
            input_size,
            hidden_size,
            input_weights,
            recurrent_weights,
            bias,
            tau,
            solver,
        })
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn step(&self, input: &[f32], state: &[f32], dt: f32) -> Result<Vec<f32>, LiquidError> {
        if input.len() != self.input_size
            || state.len() != self.hidden_size
            || !dt.is_finite()
            || dt <= 0.0
        {
            return Err(LiquidError(
                "input, state, or time step has the wrong shape/value".into(),
            ));
        }
        let field = |_: f32, h: &[f32]| -> Vec<f32> {
            (0..self.hidden_size)
                .map(|row| {
                    let input_sum = dot_row(&self.input_weights, row, self.input_size, input);
                    let recurrent = dot_row(&self.recurrent_weights, row, self.hidden_size, h);
                    (-h[row] + (input_sum + recurrent + self.bias[row]).tanh()) / self.tau[row]
                })
                .collect()
        };
        NeuralOde::new(field, self.solver).solve(state, 0.0, dt, 1)
    }
}

/// Closed-form continuous-time (CfC) cell. Unlike a discretized RNN, its gate
/// depends explicitly on elapsed time, so irregularly sampled sequences can be
/// processed without resampling.
#[derive(Debug, Clone)]
pub struct CfcCell {
    input_size: usize,
    hidden_size: usize,
    candidate_weights: Vec<f32>,
    recurrent_weights: Vec<f32>,
    decay: Vec<f32>,
}

impl CfcCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, LiquidError> {
        if input_size == 0 || hidden_size == 0 {
            return Err(LiquidError("layer dimensions must be positive".into()));
        }
        let candidate_weights = (0..hidden_size * input_size)
            .map(|i| ((i * 13 + 1) % 29) as f32 / 58.0 - 0.25)
            .collect();
        let recurrent_weights = (0..hidden_size * hidden_size)
            .map(|i| ((i * 5 + 2) % 23) as f32 / 92.0 - 0.125)
            .collect();
        Ok(Self {
            input_size,
            hidden_size,
            candidate_weights,
            recurrent_weights,
            decay: vec![1.0; hidden_size],
        })
    }

    pub fn step(
        &self,
        input: &[f32],
        state: &[f32],
        elapsed: f32,
    ) -> Result<Vec<f32>, LiquidError> {
        if input.len() != self.input_size
            || state.len() != self.hidden_size
            || !elapsed.is_finite()
            || elapsed < 0.0
        {
            return Err(LiquidError(
                "input, state, or elapsed time is invalid".into(),
            ));
        }
        Ok((0..self.hidden_size)
            .map(|row| {
                let candidate = (dot_row(&self.candidate_weights, row, self.input_size, input)
                    + dot_row(&self.recurrent_weights, row, self.hidden_size, state))
                .tanh();
                let gate = (-self.decay[row] * elapsed).exp();
                gate * state[row] + (1.0 - gate) * candidate
            })
            .collect())
    }

    pub fn forward_irregular(
        &self,
        inputs: &[Vec<f32>],
        elapsed: &[f32],
    ) -> Result<Vec<Vec<f32>>, LiquidError> {
        if inputs.len() != elapsed.len() {
            return Err(LiquidError(
                "inputs and elapsed times must have equal length".into(),
            ));
        }
        let mut state = vec![0.0; self.hidden_size];
        inputs
            .iter()
            .zip(elapsed)
            .map(|(input, dt)| {
                state = self.step(input, &state, *dt)?;
                Ok(state.clone())
            })
            .collect()
    }
}
fn dot_row(matrix: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
    matrix[row * cols..(row + 1) * cols]
        .iter()
        .zip(x)
        .map(|(a, b)| a * b)
        .sum()
}

/// A compact liquid foundation model (LFM): stacked liquid cells followed by a projection.
#[derive(Debug, Clone)]
pub struct Lfm {
    layers: Vec<LiquidCell>,
    output_size: usize,
    output_weights: Vec<f32>,
}
impl Lfm {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
    ) -> Result<Self, LiquidError> {
        if hidden_sizes.is_empty() || output_size == 0 {
            return Err(LiquidError("LFM needs hidden and output dimensions".into()));
        }
        let mut previous = input_size;
        let mut layers = Vec::new();
        for &size in hidden_sizes {
            layers.push(LiquidCell::new(previous, size)?);
            previous = size;
        }
        let output_weights = (0..output_size * previous)
            .map(|i| ((i * 7 + 1) % 17) as f32 / 34.0 - 0.25)
            .collect();
        Ok(Self {
            layers,
            output_size,
            output_weights,
        })
    }
    pub fn zero_state(&self) -> Vec<Vec<f32>> {
        self.layers
            .iter()
            .map(|l| vec![0.0; l.hidden_size()])
            .collect()
    }
    pub fn step(
        &self,
        input: &[f32],
        states: &mut [Vec<f32>],
        dt: f32,
    ) -> Result<Vec<f32>, LiquidError> {
        if states.len() != self.layers.len() {
            return Err(LiquidError("one state is required per liquid layer".into()));
        }
        let mut value = input.to_vec();
        for (layer, state) in self.layers.iter().zip(states) {
            let next = layer.step(&value, state, dt)?;
            *state = next.clone();
            value = next;
        }
        Ok((0..self.output_size)
            .map(|r| dot_row(&self.output_weights, r, value.len(), &value))
            .collect())
    }
    pub fn forward(&self, sequence: &[Vec<f32>], dt: f32) -> Result<Vec<Vec<f32>>, LiquidError> {
        let mut state = self.zero_state();
        sequence
            .iter()
            .map(|x| self.step(x, &mut state, dt))
            .collect()
    }

    /// Trains the output projection while retaining the liquid reservoir.
    /// This lightweight supervised path is useful for streaming regression
    /// and classification prototypes where the recurrent dynamics are fixed.
    pub fn fit_readout(
        &mut self,
        sequence: &[Vec<f32>],
        expected: &[Vec<f32>],
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32, LiquidError> {
        if sequence.len() != expected.len()
            || sequence.is_empty()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            return Err(LiquidError(
                "training data and learning rate are invalid".into(),
            ));
        }
        let mut states = self.zero_state();
        let mut loss = 0.0;
        for (input, target) in sequence.iter().zip(expected) {
            if target.len() != self.output_size {
                return Err(LiquidError("target has the wrong output dimension".into()));
            }
            let mut hidden = input.clone();
            for (layer, state) in self.layers.iter().zip(&mut states) {
                let next = layer.step(&hidden, state, dt)?;
                *state = next.clone();
                hidden = next;
            }
            let prediction = (0..self.output_size)
                .map(|row| dot_row(&self.output_weights, row, hidden.len(), &hidden))
                .collect::<Vec<_>>();
            for row in 0..self.output_size {
                let error = prediction[row] - target[row];
                loss += error * error;
                for col in 0..hidden.len() {
                    self.output_weights[row * hidden.len() + col] -=
                        learning_rate * 2.0 * error * hidden[col];
                }
            }
        }
        Ok(loss / (sequence.len() * self.output_size) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rk4_solves_exponential_decay() {
        let ode = NeuralOde::new(|_, y: &[f32]| vec![-y[0]], OdeSolver::RungeKutta4);
        let y = ode.solve(&[1.0], 0.0, 1.0, 10).unwrap();
        assert!((y[0] - (-1.0f32).exp()).abs() < 1e-5);
    }
    #[test]
    fn lfm_runs_a_sequence() {
        let model = Lfm::new(2, &[4, 3], 1).unwrap();
        let out = model
            .forward(&[vec![1.0, 0.0], vec![0.0, 1.0]], 0.1)
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1][0].is_finite());
    }
    #[test]
    fn cfc_supports_irregular_samples_and_lfm_trains() {
        let cell = CfcCell::new(1, 2).unwrap();
        let result = cell
            .forward_irregular(&[vec![1.0], vec![0.5]], &[0.1, 0.7])
            .unwrap();
        assert_eq!(result.len(), 2);
        let mut model = Lfm::new(1, &[3], 1).unwrap();
        let loss = model
            .fit_readout(&[vec![1.0], vec![0.5]], &[vec![1.0], vec![0.0]], 0.1, 0.01)
            .unwrap();
        assert!(loss.is_finite());
    }
}
