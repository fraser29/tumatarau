
import os
import sys
from model import train_pipeline
from validation_inference import run_validation, run_inference, run_batch_inference


def main(root_dir):

    model_path = os.path.join(root_dir, "checkpoints/unet_epoch100.pth")


    # run_validation(
    #     checkpoint_path=model_path,
    #     val_source_dir=os.path.join(root_dir,"A/val"),
    #     val_target_dir=os.path.join(root_dir,"B/val"),
    #     validation_output_dir=os.path.join(root_dir,"validation_results")
    # )

    # run_inference(
    #     model_path=model_path,
    #     single_image_path="//path/to/INFERENCE/10001.png",
    #     single_output_path="//path/to/inference_results/10001_result.png",
    # )
    
    run_batch_inference(
        model_path=model_path,
        inference_input_dir=os.path.join(root_dir,"inference"),
        inference_output_dir=os.path.join(root_dir,"inference_results"),
    )

    # train_pipeline(
    #     source_dir=os.path.join(root_dir,"A/train"),
    #     target_dir=os.path.join(root_dir,"B/train"),
    #     save_dir=os.path.join(root_dir, "checkpoints"),
    #     epochs=100,
    #     batch_size=4,
    #     lr=1e-4
    # )




if __name__ == "__main__":
    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print("#ERROR - Pass root dir to run")
        sys.exit(1)
    print(f"Running with root: {root_dir}")
    main(root_dir)