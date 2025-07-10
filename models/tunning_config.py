class TuningConfig:
    def __init__(
        self,
        layer_tuning=None,
        last_block_n=1,
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_num=128,
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="1.0",
        ffn_adapter_layernorm_option="in",
        vpt_num=10,
        d_model=768,
    ):
        """
        layer_tuning: dict, key 为 Block 的索引（从 0 开始），值为 'vpt' 或 'adapter'
        last_block_n: int, 最后一个 Block 中 adapter 或 vpt 的个数
        其他参数保留原有配置
        """
        self.layer_tuning = layer_tuning if layer_tuning is not None else {}
        self.last_block_n = last_block_n
        self.ffn_adapt = ffn_adapt
        self.ffn_option = ffn_option
        self.ffn_num = ffn_num
        self.ffn_adapter_init_option = ffn_adapter_init_option
        self.ffn_adapter_scalar = ffn_adapter_scalar
        self.ffn_adapter_layernorm_option = ffn_adapter_layernorm_option
        self.vpt_num = vpt_num
        self.d_model = d_model
        

tuning_config = TuningConfig(
        layer_tuning={
            0: "vpt",
            1: "vpt",
            2: "vpt",
            3: "vpt",
            4: "vpt",
            5: "vpt",
            6: "vpt",
            7: "vpt",
            8: "vpt",
            9: "vpt",
            10: "vpt",
            11: "vpt",  # 最后一个 Block 使用 3 个并行的 VPT
        },
        last_block_n=3,  # 最后一个 Block 中使用 3 个并行的 VPT
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_num=128,
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        d_model=768,
        ffn_adapter_layernorm_option="none",
        vpt_num=10,
    )




tuning_config = TuningConfig(
        layer_tuning={
            0: "adapter",
            1: "adapter",
            2: "adapter",
            3: "adapter",
            4: "adapter",
            5: "adapter",
            6: "adapter",
            7: "adapter",
            8: "adapter",
            9: "adapter",
            10: "adapter",
            11: "adapter",  # 最后一个 Block 使用 3 个并行的 VPT
        },
        last_block_n=3,  # 最后一个 Block 中使用 3 个并行的 VPT
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_num=128,
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        d_model=768,
        ffn_adapter_layernorm_option="none",
        vpt_num=10,
    )