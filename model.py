import numpy as np
import jax
import jax.numpy as jnp
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.block_management import Block
from thrml.factor import FactorSamplingProgram
from thrml.block_sampling import (
    BlockGibbsSpec, 
    sample_states, 
    SamplingSchedule
)
from thrml.pgm import AbstractNode

class SpinNode(AbstractNode):
    pass


class Ising:
    def __init__(self, J, h, beta=1.0, use_two_coloring=True):
        """
            J: Upper triangular coupling matrix (N, N)
            h: Field vector (N,)
            beta: Inverse temperature
        """
        self.J = jnp.array(J)
        self.h = jnp.array(h)
        self.beta = beta
        self.N = len(h)
        self.use_two_coloring = use_two_coloring
        
        self.nodes = [SpinNode() for _ in range(self.N)]
        
        self.prog = self._create_program()
    
    def _create_program(self):
        if self.use_two_coloring:
            even_nodes = [self.nodes[i] for i in range(0, self.N, 2)]
            odd_nodes = [self.nodes[i] for i in range(1, self.N, 2)]
            free_blocks = [Block(even_nodes), Block(odd_nodes)]
        else:
            free_blocks = [Block(self.nodes)]
        
        node_shape_dtypes = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        
        spec = BlockGibbsSpec(free_blocks, [], node_shape_dtypes)
        
        factors = []
        
        if jnp.any(self.h != 0):
            field_factor = SpinEBMFactor(
                [Block(self.nodes)], 
                self.beta * self.h
            )
            factors.append(field_factor)
        
        i_idx, j_idx = jnp.triu_indices(self.N, k=1)
        coupling_strengths = self.J[i_idx, j_idx]
        
        mask = coupling_strengths != 0
        if jnp.any(mask):
            i_idx_nonzero = [i_idx[k] for k in range(len(i_idx)) if mask[k]]
            j_idx_nonzero = [j_idx[k] for k in range(len(j_idx)) if mask[k]]
            coupling_strengths_nonzero = coupling_strengths[mask]
            
            coupling_factor = SpinEBMFactor(
                [Block([self.nodes[i] for i in i_idx_nonzero]),
                 Block([self.nodes[j] for j in j_idx_nonzero])],
                self.beta * coupling_strengths_nonzero
            )
            factors.append(coupling_factor)
        
        sampler = SpinGibbsConditional()
        samplers = [sampler] * len(free_blocks)
        
        prog = FactorSamplingProgram(
            gibbs_spec=spec,
            samplers=samplers,
            factors=factors,
            other_interaction_groups=[]
        )
        
        return prog
    
    def sample(self, n_chains, n_steps=1000, n_burn_in=500, key=None):
        """
            n_chains: Number of parallel chains
            n_steps: Number of samples to collect per chain
            n_burn_in: Number of burn-in steps
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        schedule = SamplingSchedule(
            n_warmup=n_burn_in,
            n_samples=n_steps,
            steps_per_sample=1
        )
        
        init_key, sample_key = jax.random.split(key)
        init_states = []
        
        for block in self.prog.gibbs_spec.free_blocks:
            block_size = len(block.nodes)
            init_states.append(
                jax.random.bernoulli(init_key, 0.5, (n_chains, block_size))
            )
            init_key, _ = jax.random.split(init_key)
        
        keys = jax.random.split(sample_key, n_chains)
        
        observe_blocks = [Block(self.nodes)]
        
        samples_list = jax.vmap(
            lambda k, *states: sample_states(
                k, self.prog, schedule, list(states), [], observe_blocks
            )
        )(keys, *init_states)
        
        samples = samples_list[0] 
        
        samples = 2 * samples.astype(jnp.int32) - 1
        
        return samples



class MaxEntBoltzmann:
    def __init__(self, n, target_s, target_ss, beta=1.0):
        """
            n: number of spins
            target_s: Target magnetizations (N,)
            target_ss: Target correlations (N, N) upper triangular
            beta: Inverse temperature
        """
        self.n = n
        self.beta = beta
        
        self.J = np.triu(np.random.normal(0, 0.05, (n, n)))
        np.fill_diagonal(self.J, 0)
        self.h = np.random.normal(0, 0.05, n)
        
        self.target_s = np.array(target_s)
        self.target_ss = np.array(target_ss)
        
        self.s_errors = []
        self.ss_errors = []
        self.J_history = []
        self.h_history = []
    
    def compute_probs(self, n_chains=500, n_steps=200, n_burn_in=100, key=None):
        """
            n_chains: Number of parallel chains (can be much larger with GPU!)
            n_steps: Number of samples per chain
            n_burn_in: Burn-in period
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        
        sampler = Ising(self.J, self.h, beta=self.beta, use_two_coloring=True)
        samples = sampler.sample(n_chains, n_steps, n_burn_in, key)
        flat_samples = samples.reshape(-1, self.n)
        
        model_s = np.array(jnp.mean(flat_samples, axis=0))
        model_ss = np.array(
            jnp.einsum('bi,bj->ij', flat_samples, flat_samples) / flat_samples.shape[0]
        )
        
        return model_s, model_ss
    
    def compute_gradients(self, model_s, model_ss):
        dh = self.target_s - model_s
        
        dJ = self.target_ss - model_ss
        dJ = np.triu(dJ, k=1)  
        
        return dh, dJ
    
    def compute_error(self, model_s, model_ss):
        s_error = np.mean(np.abs(self.target_s - model_s))
        
        mask = np.triu(np.ones((self.n, self.n)), k=1).astype(bool)
        ss_error = np.mean(np.abs(self.target_ss[mask] - model_ss[mask]))
        
        return s_error, ss_error
    
    def fit(self, 
            n_iterations=100, 
            learning_rate=0.05,
            momentum=0.9,
            n_chains=500,
            n_steps=200,
            n_burn_in=100,
            adaptive_lr=True,
            patience=5,
            lr_decay=0.7,
            min_lr=1e-5,
            verbose=True):
        """
            n_iterations: Number of gradient steps
            learning_rate: Initial learning rate
            momentum: Momentum coefficient
            n_chains: Number of sampling chains
            n_steps: Samples per chain
            n_burn_in: Burn-in period
            adaptive_lr: Whether to use adaptive learning rate
            patience: Iterations without improvement before reducing lr
            lr_decay: Learning rate decay factor
            min_lr: Minimum learning rate
            verbose: Print progress
        """
        v_h = np.zeros_like(self.h)
        v_J = np.zeros_like(self.J)
        
        best_error = float('inf')
        no_improve_count = 0
        lr = learning_rate
        
        for iteration in range(n_iterations):
            key = jax.random.PRNGKey(iteration)
            model_s, model_ss = self.compute_probs(
                n_chains=n_chains,
                n_steps=n_steps,
                n_burn_in=n_burn_in,
                key=key
            )
            
            dh, dJ = self.compute_gradients(model_s, model_ss)
            
            v_h = momentum * v_h + lr * dh
            v_J = momentum * v_J + lr * dJ
            
            self.h += v_h
            self.J += v_J
            
            self.J = np.triu(self.J, k=1)
            
            s_error = np.linalg.norm(self.target_s - model_s)
            ss_error = np.linalg.norm(self.target_ss - model_ss)
            total_error = s_error + ss_error
            
            self.s_errors.append(s_error)
            self.ss_errors.append(ss_error)
            self.h_history.append(self.h.copy())
            self.J_history.append(self.J.copy())
            
            if adaptive_lr:
                if total_error < best_error * 0.99: 
                    best_error = total_error
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                
                if no_improve_count >= patience:
                    lr = max(lr * lr_decay, min_lr)
                    no_improve_count = 0
                    if verbose:
                        print(f" reducing learning rate to {lr:.6f}")
            if verbose and (iteration % 1 == 0 or iteration == n_iterations - 1):
                print(f"Iter {iteration:3d}: "
                      f"s_err={s_error:.6f}, "
                      f"ss_err={ss_error:.6f}, "
                      f"lr_h={lr:.6f}, ")

        
        return self.J, self.h

