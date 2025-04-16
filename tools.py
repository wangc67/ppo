class UniformDecelerationTimer:
    def __init__(self):
        super().__init__()
    def __call__(self,initial_velocity, deceleration, target_position):
        """计算到达目标位置所需时间"""
        # 实现匀减速运动方程: s = ut - 0.5at²
        # 返回时间解...
        if initial_velocity < 0:
            initial_velocity = -initial_velocity
            target_position = -target_position
        if target_position < 0:
            return -1
        u = initial_velocity  # 初速度
        a = deceleration     # 减速度（应为正值）
        s = u ** 2 / 2*a
        if s < target_position:
            return -1
        else: 
            t = (u - (u ** 2 - 2 * a * target_position) ** 0.5) / a
            return t if t >= 0 else -1
        
# Timer = UniformDecelerationTimer()
# print(Timer(10, 1, 50)) 
