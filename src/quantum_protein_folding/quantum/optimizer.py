"""Classical optimizers for variational quantum algorithms."""

import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result of classical optimization.
    
    Attributes:
        optimal_value: Final objective function value
        optimal_params: Optimal parameter values
        n_iterations: Number of iterations
        n_function_evaluations: Number of objective evaluations
        convergence_history: History of objective values
        success: Whether optimization succeeded
    """
    optimal_value: float
    optimal_params: np.ndarray
    n_iterations: int
    n_function_evaluations: int
    convergence_history: List[float]
    success: bool


class VariationalOptimizer:
    """Classical optimizer for variational quantum algorithms.
    
    Wraps scipy optimizers with logging and convergence tracking.
    """
    
    def __init__(
        self,
        method: str = 'COBYLA',
        maxiter: int = 1000,
        tol: float = 1e-6,
        options: Optional[dict] = None
    ):
        """Initialize optimizer.
        
        Args:
            method: Optimization method ('COBYLA', 'SLSQP', 'L-BFGS-B', etc.)
            maxiter: Maximum iterations
            tol: Convergence tolerance
            options: Additional options passed to scipy
        """
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.options = options or {}
        
        # Tracking
        self.iteration_count = 0
        self.function_eval_count = 0
        self.convergence_history = []
    
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> OptimizationResult:
        """Run optimization.
        
        Args:
            objective_function: Function to minimize f(params) -> value
            initial_params: Initial parameter guess
            bounds: Parameter bounds [(min, max), ...]
            
        Returns:
            OptimizationResult with optimal parameters
        """
        # Reset tracking
        self.iteration_count = 0
        self.function_eval_count = 0
        self.convergence_history = []
        
        # Wrapped objective with tracking
        def tracked_objective(params):
            value = objective_function(params)
            self.function_eval_count += 1
            self.convergence_history.append(value)
            return value
        
        # Callback for iteration tracking
        def callback(params):
            self.iteration_count += 1
        
        # Run optimization
        if self.method == 'SPSA':
            # SPSA (Simultaneous Perturbation Stochastic Approximation)
            result = self._spsa_optimize(
                tracked_objective, initial_params, bounds
            )
        else:
            # Scipy optimizers
            result = minimize(
                tracked_objective,
                initial_params,
                method=self.method,
                bounds=bounds,
                tol=self.tol,
                callback=callback,
                options={'maxiter': self.maxiter, **self.options}
            )
        
        return OptimizationResult(
            optimal_value=result.fun,
            optimal_params=result.x,
            n_iterations=self.iteration_count,
            n_function_evaluations=self.function_eval_count,
            convergence_history=self.convergence_history,
            success=result.success
        )
    
    def _spsa_optimize(
        self,
        objective_function: Callable,
        initial_params: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]]
    ):
        """SPSA optimizer implementation.
        
        Simultaneous Perturbation Stochastic Approximation.
        Useful for noisy objective functions (quantum measurements).
        """
        params = initial_params.copy()
        n_params = len(params)
        
        # SPSA hyperparameters
        a = 0.16  # Step size scaling
        c = 0.1   # Perturbation size
        A = 0.1 * self.maxiter  # Stability constant
        alpha = 0.602  # Step size decay
        gamma = 0.101  # Perturbation decay
        
        best_value = objective_function(params)
        best_params = params.copy()
        
        for k in range(self.maxiter):
            # Compute step sizes
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma
            
            # Random perturbation direction
            delta = 2 * np.random.randint(0, 2, size=n_params) - 1
            
            # Evaluate at ±perturbation
            params_plus = params + ck * delta
            params_minus = params - ck * delta
            
            # Apply bounds if provided
            if bounds is not None:
                params_plus = np.clip(
                    params_plus,
                    [b[0] for b in bounds],
                    [b[1] for b in bounds]
                )
                params_minus = np.clip(
                    params_minus,
                    [b[0] for b in bounds],
                    [b[1] for b in bounds]
                )
            
            value_plus = objective_function(params_plus)
            value_minus = objective_function(params_minus)
            
            # Approximate gradient
            gradient_approx = (value_plus - value_minus) / (2 * ck * delta)
            
            # Update parameters
            params = params - ak * gradient_approx
            
            # Apply bounds
            if bounds is not None:
                params = np.clip(
                    params,
                    [b[0] for b in bounds],
                    [b[1] for b in bounds]
                )
            
            # Track best
            current_value = objective_function(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
            
            self.iteration_count += 1
        
        # Return in scipy format
        class SPSAResult:
            def __init__(self, x, fun, success=True):
                self.x = x
                self.fun = fun
                self.success = success
        
        return SPSAResult(best_params, best_value)
