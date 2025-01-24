import torch
from ultralytics import YOLO

def get_first_n_layers(model, n=23):
    """
    모델의 처음 n개 최상위 레이어를 가져옵니다.
    """
    layers = []
    for name, layer in model.named_children():
        layers.append((name, layer))
        if len(layers) >= n:
            break
    return layers

def count_parameters(layer):
    """
    레이어의 파라미터 수를 계산합니다.
    """
    return sum(p.numel() for p in layer.parameters())

def main():
    model_path = 'yolo11x.pt'  # 모델 파일 경로
    try:
        # Ultralytics YOLO 클래스를 사용하여 모델 로드
        model = YOLO(model_path)
        print("모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 모델의 실제 네트워크는 model.model 안에 있을 가능성이 높습니다.
    # 정확한 구조를 확인하기 위해 네트워크의 하위 모듈을 출력해보겠습니다.
    print("\n모델의 하위 모듈을 확인합니다:")
    for name, layer in model.model.named_children():
        print(f" - {name}: {layer.__class__.__name__}")

    # 일반적으로 model.model.model이 실제 네트워크일 가능성이 높습니다.
    # 이를 기반으로 네트워크를 설정합니다.
    if hasattr(model.model, 'model'):
        network = model.model.model
    else:
        network = model.model

    # 다시 한번 네트워크의 하위 모듈을 확인합니다.
    print("\n네트워크의 하위 모듈을 확인합니다:")
    for name, layer in network.named_children():
        print(f" - {name}: {layer.__class__.__name__}")

    # 처음 10개 최상위 레이어 가져오기
    first_10_layers = get_first_n_layers(network, n=23)

    print(f"\n모델 '{model_path}'의 1~23개 레이어 정보:\n")
    for idx, (name, layer) in enumerate(first_10_layers, start=1):
        num_params = count_parameters(layer)

        # 최상위 레이어 이름을 'model0', 'model1', 등으로 포맷팅
        # 예: name이 '0', '1' 등일 경우 'model0', 'model1'으로 표시
        if name.isdigit():
            layer_display_name = f"model {name}"
        else:
            layer_display_name = name

        layer_info = f"레이어 {idx}: {layer_display_name} - {layer.__class__.__name__}\n"
        layer_info += f"    파라미터 수: {num_params}\n"
        print(layer_info)

if __name__ == "__main__":
    main()