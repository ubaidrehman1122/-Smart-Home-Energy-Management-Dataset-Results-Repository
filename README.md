# -Smart-Home-Energy-Management-Dataset-Results-Repository
Smart Home Energy Dataset: 15-min HEMS data with RTP (€/kWh), PV/wind generation, ESS/EV SOC &amp; power, indoor/hot water temperatures, and appliance loads. Designed for PPO/DRL energy management, digital twins, and real-time optimization under uncertainty.
# 📊 Smart Home Energy Management Dataset & Results Repository

## 🏠 Repository Overview

This repository contains the complete dataset, simulation results, and visualization files supporting the research paper: **"Digital Twin-Augmented Proximal Policy Optimization for Real-Time Home Energy Management: Addressing Uncertainties in Renewables, Pricing, and Loads"** by Ubaid ur Rehman.

The dataset enables reproducibility of the Proximal Policy Optimization (PPO)-based Home Energy Management System (HEMS) with digital twin integration, validated on high-resolution real-time data.

---

## 📁 Repository Structure

```
smart-home-ems-dataset/
│
├── 📄 README.md                          # This comprehensive guide
├── 📄 LICENSE                            # MIT License
├── 📄 CITATION.cff                       # Citation file for academic use
│
├── 📂 manuscript/
│   ├── Revised_manuscript_changes_marked.docx   # Full paper with tracked changes
│   └── abstract.txt                              # Paper abstract
│
├── 📂 data/
│   ├── raw/
│   │   ├── 01_RTP_Data.csv              # Real-Time Pricing (€/kWh), 96 time steps
│   │   ├── 02_WindPV_Data.csv           # Wind & PV generation profiles (kW)
│   │   ├── 03_ESS_Data.csv              # Energy Storage System: charging, discharging, SOC
│   │   ├── 04_EV_Data.csv               # Electric Vehicle: charging, discharging, SOC, availability
│   │   ├── 05_HotWater_Data.csv         # Water heater: ambient/indoor temps, limits
│   │   ├── 06_IndoorTemp_Data.csv       # HVAC: ambient/indoor AC temps, comfort bounds
│   │   └── 07_Appliances_Data.csv       # Schedulable loads: dishwasher, washer, dryer, vacuum
│   │
│   └── processed/
│       ├── normalized_states.csv        # Normalized MDP state vectors
│       ├── action_trajectories.csv      # PPO agent action sequences
│       └── reward_signals.csv           # Per-step reward components
│
├── 📂 results/
│   ├── figures/
│   │   ├── Fig_3_Training_Convergence.png      # PPO training reward profile
│   │   ├── Fig_4_PV_Wind_Generation.png        # Renewable generation profiles
│   │   ├── Fig_5_RTP_Profile.png               # Real-time electricity pricing
│   │   ├── Fig_6_Indoor_Temperature.png        # HVAC thermal regulation
│   │   ├── Fig_7_HotWater_Temperature.png      # Water heater performance
│   │   ├── Fig_8_EV_Operation.png              # EV charge/discharge & SOC
│   │   ├── Fig_9_ESS_Operation.png             # ESS cycling & SOC evolution
│   │   ├── Fig_10_Appliance_Scheduling.png     # Interruptible load activation
│   │   ├── Fig_11_Energy_Composition.png       # Stacked energy flow visualization
│   │   ├── Fig_12_Grid_Interaction.png         # Total load & grid import/export
│   │   ├── Fig_13_Benchmark_Comparison.png     # Cost comparison vs. SOTA methods
│   │   ├── Fig_14_Sensitivity_LR.png           # Learning rate sensitivity
│   │   ├── Fig_15_Sensitivity_Gamma.png        # Discount factor sensitivity
│   │   ├── Fig_16_Sensitivity_Clip.png         # PPO clip parameter sensitivity
│   │   └── Fig_17_Sensitivity_Lambda.png       # GAE lambda sensitivity
│   │
│   └── tables/
│       ├── Table_1_Literature_Comparison.csv   # SOTA model comparison
│       ├── Table_2_Python_Libraries.csv        # Implementation dependencies
│       ├── Table_3_Hyperparameters.csv         # PPO training configuration
│       ├── Table_4_Appliance_Parameters.csv    # Component specifications
│       ├── Table_5_Performance_Metrics.csv     # Final simulation results
│       └── Table_6_Benchmark_Results.csv       # Comparative performance data
│
├── 📂 code/
│   ├── environment.py                   # Custom Gym HEMS environment
│   ├──ppo_agent.py                      # Actor-critic PPO implementation
│   ├──digital_twin.py                   # Physics-based simulation module
│   ├──train.py                          # Training pipeline (2500 episodes)
│   ├──test.py                           # Real-time inference & evaluation
│   ├──plot_utils.py                     # Visualization utilities
│   └── requirements.txt                 # Python dependencies
│
└── 📂 notebooks/
    ├── 01_data_exploration.ipynb       # Dataset inspection & statistics
    ├── 02_training_analysis.ipynb      # Reward convergence & policy diagnostics
    ├── 03_realtime_simulation.ipynb    # Reproduce Fig 4-12 results
    └── 04_sensitivity_analysis.ipynb   # Reproduce Fig 14-17 hyperparameter study
```

---

## 🗄️ Dataset Description

### 🔹 Primary Dataset: 96-Step Real-Time Simulation

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| **Temporal Resolution** | 15 minutes | - | 96 steps = 24-hour horizon |
| **Training Period** | 180 days | - | 17,280 time steps for PPO training |
| **Testing Period** | 1 day | - | Held-out 96-step real-time validation |
| **Geographic Context** | Porto, Portugal | - | Climate & pricing profile |
| **Data Source** | Physical testbed + public datasets | - | [GitHub: 6-Months-Smart-Home-Energy-Dataset](https://github.com/ubaidrehman1122/6-Months-Smart-Home-Energy-Dataset) |

### 🔹 Component Specifications

#### ⚡ Renewable Generation
| Source | Peak Power | Profile Type | Variability |
|--------|-----------|--------------|-------------|
| Photovoltaic (PV) | 6–10 kW | Diurnal (bell-shaped) | Weather-dependent irradiance |
| Wind Turbine | 2.5–4 kW | Intermittent | Sustained off-solar generation |

#### 🔋 Energy Storage Systems
| Component | Capacity | Efficiency | Power Limits | SOC Bounds |
|-----------|----------|------------|--------------|------------|
| ESS (Stationary) | 12.0 kWh | 90% round-trip | ±1.5 kW | [1.8, 10.2] kWh |
| EV Battery | 30.0 kWh | 90% round-trip | ±2.5 kW | [6.0, 27.0] kWh |
| *EV Availability* | — | — | — | Absent: steps 32–68 |

#### 🌡️ Thermal Systems
| System | Comfort Range | Setpoint | Max Power | Dynamics |
|--------|--------------|----------|-----------|----------|
| HVAC (Indoor) | [22, 26] °C | 24 °C | 2.5 kW | First-order thermal inertia |
| Water Heater | [40, 46] °C | 43 °C | 2.5 kW | Stratification-aware model |

#### 🔌 Load Profiles
| Category | Appliances | Power Range | Scheduling |
|----------|-----------|-------------|------------|
| Non-Interruptible | Lights, Refrigerator, Stove, Kettle, Iron, TV, Fan, Microwave, Toaster | 0.08–4.5 kW | User-defined, exogenous |
| Interruptible (Schedulable) | Dishwasher (2.0 kW, 4 steps), Washing Machine (1.5 kW, 5 steps), Dryer (1.1 kW, 3 steps), Vacuum (1.2 kW, 3 steps) | 1.1–2.0 kW | PPO-optimized, mutual exclusivity enforced |

#### 💰 Pricing & Grid
| Parameter | Value | Notes |
|-----------|-------|-------|
| Real-Time Price (RTP) | 0.12–0.24 €/kWh | Peak: steps 25–80 (≥0.15 €/kWh) |
| Export Tariff | 70% of import price | Incentivizes self-consumption |
| Grid Constraints | Unlimited import/export | No capacity limits in simulation |

---

## 📈 Key Results Summary

### ✅ Performance Metrics (96-Step Test)
| Metric | Value | Unit | Significance |
|--------|-------|------|-------------|
| **Total Optimized Cost** | 13.53 | € | 42.8% reduction vs. baseline |
| **Baseline Cost** | 23.68 | € | No optimization, no ESS/EV |
| **Cost Savings** | 10.15 | € | Absolute improvement |
| **Inference Time** | 125.46 | ms | Real-time capable (<150 ms) |
| **Indoor Comfort Compliance** | 1.00 | fraction | Zero violations of [22, 26]°C |
| **Hot Water Comfort Compliance** | 1.00 | fraction | Zero violations of [40, 46]°C |
| **Overall Constraint Satisfaction** | 100% | — | SOC, appliance logic, thermal bounds |

### 🏆 Benchmark Comparison
| Model | Cost (€) | Savings vs. Baseline | Inference (ms) | Comfort |
|-------|----------|---------------------|----------------|---------|
| Baseline (No Opt.) | 23.68 | 0% | — | 1.00 |
| Uncertainty-Aware PPO [42] | 14.56 | 38.5% | 150 | 1.00 |
| DRL PEV Scheduling [44] | 18.92 | 20.1% | 180 | 0.95 |
| ADP Real-Time HEMS [45] | 17.45 | 26.3% | 200 | 0.98 |
| Robust MILP [46] | 16.78 | 29.1% | 500 | 1.00 |
| MAS Self-Approaching [47] | 19.34 | 18.3% | 300 | 0.97 |
| **Proposed DT-PPO** | **13.53** | **42.8%** | **125** | **1.00** |

### 🔍 Sensitivity Analysis Highlights
| Hyperparameter | Optimal Value | Cost Range (± variation) | Comfort Stability |
|----------------|--------------|-------------------------|------------------|
| Learning Rate (α) | 3×10⁻⁴ | 13.53–16.45 € | ≥0.95 |
| Discount Factor (γ) | 0.99 | 13.49–16.78 € | ≥0.97 |
| Clip Parameter (ε) | 0.2 | 13.53–14.10 € | ≥0.99 |
| GAE Lambda (λ) | 0.95 | 13.53–14.65 € | ≥0.98 |
| Episode Length | 96 steps | 13.49–17.33 € | ≥0.96 |

*All variations maintain real-time inference capability and zero hard-constraint violations.*

---

## 🛠️ Usage Instructions

### 🔧 Prerequisites
```bash
Python ≥ 3.9
TensorFlow ≥ 2.12
TensorFlow Probability ≥ 0.20
pandas, numpy, matplotlib, gym
```

### 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/smart-home-ems-dataset.git
cd smart-home-ems-dataset

# Install dependencies
pip install -r code/requirements.txt

# Explore data
python -m notebooks.01_data_exploration

# Reproduce main results
python code/test.py --config configs/realtime_96step.yaml

# Run sensitivity analysis
python -m notebooks.04_sensitivity_analysis
```

### 📊 Reproducing Figures
All figures (Fig 3–17) can be regenerated using:
```bash
python code/plot_utils.py --figure Fig_13 --output results/figures/
```

### 🧪 Custom Simulations
Modify `configs/custom_scenario.yaml` to:
- Adjust renewable capacity scaling (±50%)
- Change comfort bounds or appliance schedules
- Test alternative pricing profiles
- Evaluate different ESS/EV sizing

---

## 📚 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{rehman2026digital,
  title={Digital Twin-Augmented Proximal Policy Optimization for Real-Time Home Energy Management: Addressing Uncertainties in Renewables, Pricing, and Loads},
  author={Rehman, Ubaid ur},
  year={2026},
  url={https://github.com/yourusername/smart-home-ems-dataset}
}
```

Also cite the foundational works:
```bibtex
@article{rehman2025ppo,
  title={Proximal Policy Optimization–Driven Real-Time Home Energy Management System with Storage and Renewables},
  author={Rehman, Ubaid ur},
  journal={Process Integration and Optimization for Sustainability},
  volume={9},
  pages={507--536},
  year={2025},
  doi={10.1007/s41660-024-00476-6}
}

@article{rehman2025gui,
  title={A GUI application for real-time home energy management using uncertainty-aware Proximal Policy Optimization with energy storage, electric vehicle, and renewables},
  author={Rehman, Ubaid ur},
  journal={Journal of Building Engineering},
  volume={111},
  pages={113174},
  year={2025},
  doi={10.1016/j.jobe.2025.113174}
}
```

---

## ⚖️ License & Ethics

- **License**: MIT License — free for academic and commercial use with attribution
- **Ethics Approval**: Not required (no human/animal subjects)
- **Data Privacy**: All data is anonymized, synthetic, or publicly sourced; no personal information included
- **Conflict of Interest**: None declared

---

## 🤝 Contributing & Support

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For issues or questions:
- 🐛 Bug reports: Use GitHub Issues with `[BUG]` tag
- 💡 Feature requests: Use GitHub Issues with `[ENHANCEMENT]` tag
- 📧 General inquiries: Contact ubaid@isep.ipp.pt

---

## 🔗 Related Resources

- 🌐 [Digital Twin Concepts for Energy Systems](https://www.sciencedirect.com/topics/engineering/digital-twin)
- 📚 [Proximal Policy Optimization Original Paper](https://arxiv.org/abs/1707.06347)
- 🗃️ [Open Energy Modelling Initiative (openmod-initiative.org)](https://openmod-initiative.org)
- 📊 [IEEE PES Smart Grid Data Repository](https://ieee-pes.org/publications/data-sets)

---

> **Disclaimer**: This dataset is provided "as is" for research and educational purposes. The authors make no warranties regarding accuracy, completeness, or fitness for a particular purpose. Users are responsible for validating results in their specific application contexts.

*Last updated: May 2026 | Maintained by Ubaid ur Rehman, Polytechnic of Porto (ISEP/IPP), Portugal* 🇵🇹
