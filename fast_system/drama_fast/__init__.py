"""DRAMA-X Fast System (Phase-1).

目标：先把“快系统”端到端闭环跑通：真实 T 帧输入 -> (Backbone) -> (Single-Query Joint Head) -> bbox + risk。
慢系统(LLM)的先验接口只留坑位，不参与 Phase-1 训练。
"""
