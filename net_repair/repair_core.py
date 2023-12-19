import numpy as np
import cvxpy as cp
import copy


class Repair:
    """
    R = Repair(dx_set, epsilon, theta)
    R.add_constraints([dA_lb, dA_ub, bias_lb, bias_ub]) # 
    R.repair(step_num=100)
    print(R.delta_theta)
    
    dx_set: the linear constraints that dx should satisfy. dx_set = [A, b] <=> Ax<=b.
    epsilon: the epsilon bound for the property.
    theta: DNN’s weight in the last layer.
    p: we optimize the p norm for \Delta theta. 
    """
    def __init__(self, dx_set, epsilon, theta, p=2, print_process=False):
        self.x_dim = dx_set[0].shape[1]
        self.z_dim = theta.shape[1]
        self.out_dim = theta.shape[0]
        self.epsilon = epsilon
        self.dx_set = dx_set  # [A, b]: Ax<=b
        self.theta = theta
        self.p = p
        self.reset_constraints()
        self.print_process = print_process
        
    def reset_delta_theta(self):
        self.delta_theta = -self.theta
        self.prev_delta_theta = None
        self.gradient = None
        self.Hessian = None
    
    def reset_constraints(self):
        self.repair_constraints = []
        self.reset_delta_theta()

    def repair(self, step_num, step_size=0.3, tol=0.00005, t=1, beta=[0.9, 1.1]):
        """
        tol: a stopping criterion. 
        t: parameter in equation (7).
        beta: the adaptive parameters for step_size and t. At the end of every step: step_size *= beta[0], t *= beta[1]
        """
        self.curr_step = 0
        self.step_size = step_size
        self.tol = tol
        self.t = t
        self.beta = beta
        self.step_num = step_num
        while self.curr_step < self.step_num:
            self.is_retreat = False
            self.update()
        self.check()
        return self.delta_theta

    def update(self):
        self.update_gradient()
        d = self.step_size*self.gradient
        # d = np.matmul(np.linalg.inv(self.Hessian), self.gradient).reshape([self.out_dim, self.z_dim])
        if np.linalg.norm(d) < self.tol:
            if self.print_process:
                print('Minimal step tolerate has been reached, total number of step we took: ', self.curr_step+1)
            self.curr_step = self.step_num
        if not self.is_retreat:  ### If there is a violation of the constraints, we do not update delta_theta. Instead, we increase self.t to have a stronger barrier.
            if self.print_process:
                # print('Current delta_theta:', self.delta_theta)
                print('Current delta_theta norm:', np.linalg.norm(self.delta_theta, self.p))
                print('Distance to boundary:', self.dis_to_boundary)
            self.prev_delta_theta = copy.deepcopy(self.delta_theta)
            self.delta_theta -= d
            self.curr_step += 1
            self.t *= self.beta[1]
        self.step_size *= self.beta[0]

    
    def retreat(self):      ### We increase self.t to have a stronger barrier.
        self.delta_theta = self.prev_delta_theta
        self.t = self.t / (self.beta[1])*10
        self.is_retreat = True
        
    def check(self):
        for i in range(len(self.repair_constraints)):
            opti_epsilon, e_j, opti_dx, opti_dz = self.solve_convex(i)
            if opti_epsilon > self.epsilon:
                self.retreat()
                # print('Find violation on constraints! Repair fail!')  ### If it's violate any constraints, we retreat to the last delta_theta that always satisfy the constraints.

    def add_constraints(self, constraints):
        dA_lb, dA_ub, bias_lb, bias_ub = constraints
        self.repair_constraints.append([dA_lb, dA_ub, bias_lb, bias_ub])

    def solve_convex(self, i):
        dA_lb, dA_ub, bias_lb, bias_ub = self.repair_constraints[i]
        dx = cp.Variable(self.x_dim)
        dz = cp.Variable(self.z_dim)
        A, b = self.dx_set
        res = []
        for j, sign in zip(range(self.out_dim), [-1.0, 1.0]):
            e_j = np.zeros(self.out_dim)
            e_j[j] = sign
            cost = e_j @ (self.theta+self.delta_theta) @ dz
            prob = cp.Problem(cp.Maximize(cost), [A @ dx <= b,
                                                  dA_lb @ dx + bias_lb <= dz,
                                                  dz <= dA_ub @ dx + bias_ub])
            prob.solve()
            if prob.value > self.epsilon:
                self.retreat()
            if len(res) == 0 or prob.value > res[0]:
                res = [prob.value, e_j, dx.value, dz.value]
        return res
    
    def update_gradient(self):
        flatten_delta_theta = self.delta_theta
        delta_theta_norm = np.linalg.norm(flatten_delta_theta, self.p)
        self.gradient = delta_theta_norm**(1-self.p)*np.power(flatten_delta_theta, self.p-1)
        # self.Hessian = (self.p-1)*self.p*np.diag(np.power(flatten_delta_theta, self.p-2))
        dis_to_boundary = None
        for i in range(len(self.repair_constraints)):
            if self.is_retreat:
                break
            opti_epsilon, e_j, opti_dx, opti_dz = self.solve_convex(i)
            if dis_to_boundary is None or dis_to_boundary > self.epsilon - opti_epsilon:
                dis_to_boundary = self.epsilon - opti_epsilon
            temp = np.tensordot(np.expand_dims(e_j, 0), np.expand_dims(opti_dz, 0), axes=[0, 0])
            self.gradient -= 1/self.t * 1/(self.epsilon - opti_epsilon) * temp
            # self.Hessian += 1/self.t * (self.epsilon - opti_epsilon)**(-2) * np.tensordot(np.expand_dims(temp, 0), np.expand_dims(temp, 0), axes=[0, 0])
        if not self.is_retreat:
            self.dis_to_boundary = dis_to_boundary
            
        
if __name__ == '__main__':
    in_dim, z_dim, out_dim = 2, 3, 5
    in_costraints = 4
    dx_set = [np.block([[np.eye(in_dim)], [-np.eye(in_dim)]]), 
                np.block([np.ones(in_dim), np.zeros(in_dim)])]
    print(dx_set[0].shape, dx_set[1].shape)
    epsilon = 1
    theta = np.random.random([out_dim, z_dim])
    R = Repair(dx_set, epsilon, theta, print_process=True)
    for i in range(10):
        dA_lb, bias_lb = np.random.random([z_dim, in_dim]), np.random.random(z_dim)
        dA_ub, bias_ub = dA_lb + np.random.random(dA_lb.shape), np.ones_like(bias_lb)
        R.add_constraints([dA_lb, dA_ub, bias_lb, bias_ub])
    R.repair(100)

    # in_dim = 3
    # z_dim = 1
    # epsilon = 0.3
    # theta = np.arange(10, 12).reshape([2, 1])*1.0
    # dx_set = [np.block([[np.eye(in_dim)], [-np.eye(in_dim)]]), 
    #              np.block([np.ones(in_dim), np.zeros(in_dim)])]
    # R = Repair(dx_set, epsilon, theta, print_process=True)
    # epsilon = 0.1
    # dA_lb, bias_lb = np.zeros([z_dim, in_dim]), -np.ones(z_dim)*0.1
    # dA_ub, bias_ub = np.zeros([z_dim, in_dim]), np.ones(z_dim)*0.1
    # R.add_constraints([dA_lb, dA_ub, bias_lb, bias_ub])
    # R.repair(300)
    
    
    
    
    
    
    
    
    
    
    
    
    
    