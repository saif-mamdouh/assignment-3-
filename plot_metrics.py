import json
import matplotlib.pyplot as plt

def plot_phase():
    # Load metrics for both phases
    with open("metrics_phase1.json", "r") as f1, open("metrics_phase2.json", "r") as f2:
        m1 = json.load(f1)
        m2 = json.load(f2)

    # Extract epoch numbers
    epochs1 = sorted(int(k) for k in m1.keys())
    epochs2 = sorted(int(k) for k in m2.keys())

    # Extract values
    train1 = [m1[str(e)]["train_loss"] for e in epochs1]
    train2 = [m2[str(e)]["train_loss"] for e in epochs2]
    val1 = [m1[str(e)]["val_loss"] for e in epochs1]
    val2 = [m2[str(e)]["val_loss"] for e in epochs2]
    iou1 = [m1[str(e)]["iou"] for e in epochs1]
    iou2 = [m2[str(e)]["iou"] for e in epochs2]

    # ---------- Plot 1: Training Loss ----------
    plt.figure(figsize=(7,5))
    plt.plot(epochs1, train1, marker='o', label="Train Loss (Phase 1)")
    plt.plot(epochs2, train2, marker='x', label="Train Loss (Phase 2)")
    plt.title("Training Loss — Phase 1 vs Phase 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss_phase1_2.png", dpi=150)
    plt.close()

    # ---------- Plot 2: Validation Loss ----------
    plt.figure(figsize=(7,5))
    plt.plot(epochs1, val1, marker='o', label="Validation Loss (Phase 1)")
    plt.plot(epochs2, val2, marker='x', label="Validation Loss (Phase 2)")
    plt.title("Validation Loss — Phase 1 vs Phase 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_loss_phase1_2.png", dpi=150)
    plt.close()

    # ---------- Plot 3: IoU ----------
    plt.figure(figsize=(7,5))
    plt.plot(epochs1, iou1, marker='o', label="IoU (Phase 1)")
    plt.plot(epochs2, iou2, marker='x', label="IoU (Phase 2)")
    plt.title("IoU — Phase 1 vs Phase 2")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig("iou_phase1_2.png", dpi=150)
    plt.close()

    print("✅ Saved: train_loss_phase1_2.png, val_loss_phase1_2.png, iou_phase1_2.png")

if __name__ == "_main_":
    plot_phase()