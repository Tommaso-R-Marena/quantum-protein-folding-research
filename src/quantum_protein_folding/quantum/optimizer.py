"""Classical optimizers for variational quantum algorithms.

Wraps various optimization methods for use in VQE and QAOA.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import warnings
    warnings.warn("SciPy not available. Some optimizers will not work.")


class OptimizerType(Enum):
    """Supported optimizer types."""
    COBYLA = "COBYLA"  # Constrained Optimization BY Linear Approximation
    SPSA = "SPSA"  # Simultaneous Perturbation Stochastic Approximation
    LBFGSB = "L-BFGS-B"  # Limited-memory BFGS with bounds
    SLSQP = "SLSQP"  # Sequential Least Squares Programming
    NELDER_MEAD = "Nelder-Mead"  # Simplex algorithm
    POWELL = "Powell"  # Powell's method
    ADAM = "ADAM"  # Adaptive Moment Estimation
    GRADIENT_DESCENT = "GradientDescent"  # Basic gradient descent


@dataclass
class OptimizationResult:
    """Result from optimization.
    
    Attributes:
        optimal_params: Best parameter values found
        optimal_value: Best function value
        n_iterations: Number of iterations performed
        n_evaluations: Number of function evaluations
        success: Whether optimization succeeded
        message: Termination message
        history: Optimization history (optional)
    """
    optimal_params: np.ndarray
    optimal_value: float
    n_iterations: int
    n_evaluations: int
    success: bool
    message: str
    history: Optional[Dict[str, List]] = None


class ClassicalOptimizer:
    """Wrapper for classical optimization algorithms.
    
    Provides unified interface for various optimizers used in
    variational quantum algorithms.
    
    Attributes:
        optimizer_type: Type of optimizer
        maxiter: Maximum iterations
        tol: Convergence tolerance
        options: Additional optimizer-specific options
    
    Example:
        >>> def cost_function(params):
        ...     return np.sum(params**2)
        >>> optimizer = ClassicalOptimizer(OptimizerType.COBYLA, maxiter=100)
        >>> result = optimizer.minimize(cost_function, x0=np.random.random(5))
        >>> print(f"Minimum: {result.optimal_value:.4f}")
    """
    
    def __init__(
        self,
        optimizer_type: OptimizerType,
        maxiter: int = 1000,
        tol: float = 1e-6,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize optimizer.
        
        Args:
            optimizer_type: Type of optimization algorithm
            maxiter: Maximum number of iterations
            tol: Convergence tolerance
            options: Additional options passed to optimizer
        """
        self.optimizer_type = optimizer_type
        self.maxiter = maxiter
        self.tol = tol
        self.options = options or {}
        
        # History tracking
        self._history: Dict[str, List] = {
            'values': [],
            'params': [],
            'iterations': []
        }
        self._iteration = 0
    
    def minimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Minimize cost function.
        
        Args:
            cost_function: Function to minimize
            x0: Initial parameter values
            bounds: Optional parameter bounds
            callback: Optional callback after each iteration
        
        Returns:
            OptimizationResult object
        
        Raises:
            ValueError: If optimizer requires unavailable dependencies
        """
        self._history = {'values': [], 'params': [], 'iterations': []}
        self._iteration = 0
        
        # Wrap cost function to track history
        def tracked_cost(params):
            value = cost_function(params)
            self._history['values'].append(value)
            self._history['params'].append(params.copy())
            self._history['iterations'].append(self._iteration)
            self._iteration += 1
            
            if callback is not None:
                callback(params, value, self._iteration)
            
            return value
        
        # Run optimizer
        if self.optimizer_type in [OptimizerType.COBYLA, OptimizerType.LBFGSB,
                                   OptimizerType.SLSQP, OptimizerType.NELDER_MEAD,
                                   OptimizerType.POWELL]:
            result = self._scipy_minimize(tracked_cost, x0, bounds)
        
        elif self.optimizer_type == OptimizerType.SPSA:
            result = self._spsa_minimize(tracked_cost, x0, bounds)
        
        elif self.optimizer_type == OptimizerType.ADAM:
            result = self._adam_minimize(tracked_cost, x0, bounds)
        
        elif self.optimizer_type == OptimizerType.GRADIENT_DESCENT:
            result = self._gradient_descent_minimize(tracked_cost, x0, bounds)
        
        else:
            raise ValueError(f"Optimizer {self.optimizer_type} not implemented")
        
        # Add history to result
        result.history = self._history.copy()
        
        return result
    
    def _scipy_minimize(
        self,
        cost_function: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]]
    ) -> OptimizationResult:
        """Use SciPy minimize."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for this optimizer")
        
        options = self.options.copy()
        options['maxiter'] = self.maxiter
        
        result = minimize(
            cost_function,
            x0,
            method=self.optimizer_type.value,
            bounds=bounds,
            tol=self.tol,
            options=options
        )
        
        return OptimizationResult(
            optimal_params=result.x,
            optimal_value=result.fun,
            n_iterations=result.nit if hasattr(result, 'nit') else len(self._history['values']),
            n_evaluations=result.nfev,
            success=result.success,
            message=result.message
        )
    
    def _spsa_minimize(
        self,
        cost_function: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]]
    ) -> OptimizationResult:
        """SPSA optimization (gradient-free).
        
        Reference:
            Spall, J. C. (1998). Implementation of the simultaneous
            perturbation algorithm for stochastic optimization.
            IEEE Transactions on Aerospace and Electronic Systems, 34(3), 817-823.
        """
        params = x0.copy()
        best_value = float('inf')
        best_params = params.copy()
        
        # SPSA parameters
        a = self.options.get('a', 0.16)  # Step size scaling
        c = self.options.get('c', 0.1)   # Perturbation size
        A = self.options.get('A', self.maxiter * 0.1)  # Stability constant
        alpha = self.options.get('alpha', 0.602)  # Step size decay
        gamma = self.options.get('gamma', 0.101)  # Perturbation decay
        
        for k in range(self.maxiter):
            # Adaptive step sizes
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma
            
            # Random perturbation
            delta = 2 * np.random.randint(0, 2, size=len(params)) - 1
            
            # Simultaneous perturbation
            params_plus = params + ck * delta
            params_minus = params - ck * delta
            
            # Apply bounds if specified
            if bounds is not None:
                params_plus = np.clip(params_plus, [b[0] for b in bounds], [b[1] for b in bounds])
                params_minus = np.clip(params_minus, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Evaluate
            y_plus = cost_function(params_plus)
            y_minus = cost_function(params_minus)
            
            # Gradient estimate
            gradient_estimate = (y_plus - y_minus) / (2 * ck * delta)
            
            # Update
            params = params - ak * gradient_estimate
            
            # Apply bounds
            if bounds is not None:
                params = np.clip(params, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Track best
            current_value = cost_function(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
            
            # Check convergence
            if k > 10 and abs(current_value - best_value) < self.tol:
                break
        
        return OptimizationResult(
            optimal_params=best_params,
            optimal_value=best_value,
            n_iterations=k + 1,
            n_evaluations=2 * (k + 1),
            success=True,
            message="SPSA optimization completed"
        )
    
    def _adam_minimize(
        self,
        cost_function: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]]
    ) -> OptimizationResult:
        """ADAM optimizer (requires gradient)."""
        # Simplified ADAM with finite differences for gradient
        params = x0.copy()
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        
        # ADAM hyperparameters
        lr = self.options.get('learning_rate', 0.01)
        beta1 = self.options.get('beta1', 0.9)
        beta2 = self.options.get('beta2', 0.999)
        epsilon = self.options.get('epsilon', 1e-8)
        
        best_value = float('inf')
        best_params = params.copy()
        
        for t in range(1, self.maxiter + 1):
            # Finite difference gradient
            grad = self._finite_difference_gradient(cost_function, params)
            
            # Update biased moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Apply bounds
            if bounds is not None:
                params = np.clip(params, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Track best
            current_value = cost_function(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
        
        return OptimizationResult(
            optimal_params=best_params,
            optimal_value=best_value,
            n_iterations=self.maxiter,
            n_evaluations=self.maxiter * (len(params) + 1),
            success=True,
            message="ADAM optimization completed"
        )
    
    def _gradient_descent_minimize(
        self,
        cost_function: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]]
    ) -> OptimizationResult:
        """Basic gradient descent with finite differences."""
        params = x0.copy()
        lr = self.options.get('learning_rate', 0.01)
        
        best_value = float('inf')
        best_params = params.copy()
        
        for _ in range(self.maxiter):
            grad = self._finite_difference_gradient(cost_function, params)
            params = params - lr * grad
            
            if bounds is not None:
                params = np.clip(params, [b[0] for b in bounds], [b[1] for b in bounds])
            
            current_value = cost_function(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
        
        return OptimizationResult(
            optimal_params=best_params,
            optimal_value=best_value,
            n_iterations=self.maxiter,
            n_evaluations=self.maxiter * (len(params) + 1),
            success=True,
            message="Gradient descent completed"
        )
    
    def _finite_difference_gradient(
        self,
        func: Callable,
        x: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient using finite differences.
        
        Args:
            func: Function to differentiate
            x: Point at which to evaluate gradient
            epsilon: Finite difference step size
        
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(x)
        f_x = func(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            grad[i] = (func(x_plus) - f_x) / epsilon
        
        return grad
