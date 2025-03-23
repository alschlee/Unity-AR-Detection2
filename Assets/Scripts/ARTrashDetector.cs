using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Barracuda;
using UnityEngine.UI;
using System.Collections.Generic;

public class ARTrashDetector : MonoBehaviour
{
    public ARCameraManager arCameraManager;
    public NNModel onnxModelAsset;
    public RawImage debugRawImage;
    public Text debugTextPrefab;
    public Canvas uiCanvas;
    
    public float confidenceThreshold = 0.6f;
    public float maxBoundingBoxDistance = 5.0f;
    public float iouThreshold = 0.4f;

    private Model runtimeModel;
    private IWorker worker;
    private const int IMG_WIDTH = 224;
    private const int IMG_HEIGHT = 224;
    private const int GRID_SIZE = 7;
    private const int NUM_BOXES = 2;
    private const int NUM_CLASSES = 18;
    
    private List<GameObject> boundingBoxes = new List<GameObject>();
    private List<Text> classTexts = new List<Text>();

    //
    private Vector3[] viewportCorners = new Vector3[4];
    
    private readonly string[] trashCategories = {
        "Aluminium foil", "Bottle cap", "Bottle", "Broken glass", "Can", 
        "Carton", "Cigarette", "Cup", "Lid", "Other litter", 
        "Other plastic", "Paper", "Plastic bag - wrapper", "Plastic container", 
        "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter"
    };

    private Dictionary<string, Color> trashClassColors = new Dictionary<string, Color>()
    {
        { "Aluminium foil", new Color(0.8f, 0.8f, 0.8f) },
        { "Bottle cap", new Color(0.0f, 0.5f, 1.0f) },
        { "Bottle", new Color(0.0f, 0.7f, 0.9f) },
        { "Broken glass", new Color(0.9f, 0.9f, 1.0f) },
        { "Can", new Color(0.7f, 0.7f, 0.7f) },
        { "Carton", new Color(0.8f, 0.5f, 0.2f) },
        { "Cigarette", new Color(1.0f, 0.6f, 0.6f) },
        { "Cup", new Color(1.0f, 0.0f, 0.0f) },
        { "Lid", new Color(0.0f, 0.8f, 0.8f) },
        { "Other litter", new Color(0.5f, 0.5f, 0.5f) },
        { "Other plastic", new Color(1.0f, 1.0f, 0.0f) },
        { "Paper", new Color(1.0f, 1.0f, 0.8f) },
        { "Plastic bag - wrapper", new Color(1.0f, 0.0f, 1.0f) },
        { "Plastic container", new Color(0.0f, 1.0f, 0.5f) },
        { "Pop tab", new Color(0.6f, 0.3f, 0.1f) },
        { "Straw", new Color(1.0f, 0.6f, 0.0f) },
        { "Styrofoam piece", new Color(1.0f, 1.0f, 1.0f) },
        { "Unlabeled litter", new Color(0.4f, 0.4f, 0.4f) }
    };

    // YOLO 출력에서 추출한 객체 정보를 담을 클래스
    private class Detection
    {
        public float x, y, width, height;
        public float confidence;
        public int classIndex;
        public string className;
        public Color color;
    }

    void Start()
{
    if (uiCanvas != null)
    {
        Camera arCam = arCameraManager.GetComponent<Camera>();
        uiCanvas.worldCamera = arCam;
        Debug.Log("uiCanvas의 Render Camera로 " + arCam.name + " 할당됨");
    }
    else
    {
        Debug.LogError("uiCanvas가 null입니다!");
    }

    if (arCameraManager == null)
    {
        Debug.LogError("[ARTrashDetector] arCameraManager가 null입니다!");
        return;
    }
    
    Debug.Log("[ARTrashDetector] arCameraManager 할당됨: " + arCameraManager.name);

    // Barracuda 모델 로드
    runtimeModel = ModelLoader.Load(onnxModelAsset);
    worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
    Debug.Log("ONNX 모델 로드 완료");
}


    void OnEnable()
    {
        if (arCameraManager == null)
        {
            Debug.LogError("ARCameraManager가 할당되지 않았습니다");
            return;
        }

        arCameraManager.frameReceived += OnCameraFrameReceived;
        Debug.Log("AR 카메라 프레임 수신 이벤트 등록 완료");
    }

    void OnDisable()
    {
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived -= OnCameraFrameReceived;
        }
        
        worker?.Dispose();
        Debug.Log("워커 리소스 해제");
        
        // 모든 바운딩 박스 제거
        ClearBoundingBoxes();
    }
    
    void OnDestroy()
    {
        worker?.Dispose();
        ClearBoundingBoxes();
    }

    void GetViewportCorners()
    {
        Camera.main.CalculateFrustumCorners(new Rect(0, 0, 1, 1), 0, Camera.MonoOrStereoscopicEye.Mono, viewportCorners);
        foreach (var corner in viewportCorners)
        {
            Debug.Log($"Viewport Corner: {corner}");
        }
    }
    
    private void ClearBoundingBoxes()
    {
        foreach (var box in boundingBoxes)
        {
            if (box != null)
                Destroy(box);
        }
        boundingBoxes.Clear();
        
        foreach (var text in classTexts)
        {
            if (text != null)
                Destroy(text.gameObject);
        }
        classTexts.Clear();
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        XRCpuImage cpuImage;
        if (!arCameraManager.TryAcquireLatestCpuImage(out cpuImage))
        {
            Debug.LogError("AR 프레임을 가져오는 데 실패했습니다.");
            return;
        }

        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
            outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
            outputFormat = TextureFormat.RGB24,
            transformation = XRCpuImage.Transformation.MirrorY
        };

        Texture2D texture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGB24, false);
        var rawTextureData = texture.GetRawTextureData<byte>();
        try
        {
            cpuImage.Convert(conversionParams, rawTextureData);
            texture.Apply();
        }
        catch (System.Exception e)
        {
            Debug.LogError("이미지 변환 중 오류 발생: " + e.Message);
            cpuImage.Dispose();
            return;
        }
        finally
        {
            cpuImage.Dispose();
        }

        // 이미지 크기 조정
        Texture2D resized = ScaleTexture(texture, IMG_WIDTH, IMG_HEIGHT);

        if (debugRawImage != null)
            debugRawImage.texture = resized;

        // 모델 추론 실행
        using (Tensor input = new Tensor(resized, channels: 3))
        {
            try
            {
                worker.Execute(input);
                Tensor output = worker.PeekOutput("output");
                
                // 기존 바운딩 박스 삭제
                ClearBoundingBoxes();
                
                // 출력 처리
                List<Detection> detections = ProcessYoloOutput(output);
                
                // NMS(Non-Maximum Suppression) 적용
                detections = ApplyNMS(detections);
                
                // 바운딩 박스 표시
                foreach (var detection in detections)
                {
                    CreateBoundingBox(detection);
                }
                
                output.Dispose();
            }
            catch (System.Exception e)
            {
                Debug.LogError("모델 추론 중 오류 발생: " + e.Message);
            }
        }

        Destroy(texture);
        Destroy(resized);
    }
    
    private Texture2D ScaleTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
        Graphics.Blit(source, rt);
        
        RenderTexture previousRT = RenderTexture.active;
        RenderTexture.active = rt;
        
        Texture2D result = new Texture2D(targetWidth, targetHeight);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();
        
        RenderTexture.active = previousRT;
        RenderTexture.ReleaseTemporary(rt);
        
        return result;
    }
    
    private List<Detection> ProcessYoloOutput(Tensor output)
    {
        List<Detection> detections = new List<Detection>();
        
        // 출력 형식: [배치, 높이, 너비, 채널]
        // 채널 = 5 * NUM_BOXES + NUM_CLASSES (5 = (x, y, w, h, confidence))
        for (int cy = 0; cy < GRID_SIZE; cy++)
        {
            for (int cx = 0; cx < GRID_SIZE; cx++)
            {
                for (int b = 0; b < NUM_BOXES; b++)
                {
                    int offset = 5 * b;
                    float confidence = output[0, cy, cx, offset + 4];
                    
                    if (confidence > confidenceThreshold)
                    {
                        float x = (output[0, cy, cx, offset] + cx) / GRID_SIZE;
                        float y = (output[0, cy, cx, offset + 1] + cy) / GRID_SIZE;
                        float width = Mathf.Exp(output[0, cy, cx, offset + 2]) / GRID_SIZE;
                        float height = Mathf.Exp(output[0, cy, cx, offset + 3]) / GRID_SIZE;
                        
                        int classOffset = 5 * NUM_BOXES;
                        float maxClassProb = 0;
                        int maxClassIndex = 0;
                        
                        for (int c = 0; c < NUM_CLASSES; c++)
                        {
                            float classProb = output[0, cy, cx, classOffset + c];
                            if (classProb > maxClassProb)
                            {
                                maxClassProb = classProb;
                                maxClassIndex = c;
                            }
                        }
                        
                        // 최종 신뢰도 = 객체 존재 확률 * 클래스 확률
                        float finalConfidence = confidence * maxClassProb;
                        
                        // Confidence 값 디버깅 로그
                        Debug.Log($"Confidence: {confidence:F3}, ClassProb: {maxClassProb:F3}, FinalConfidence: {finalConfidence:F3}");
                        
                        if (finalConfidence > confidenceThreshold)
                        {
                            string className = trashCategories[maxClassIndex];
                            Color color = trashClassColors.ContainsKey(className) ? trashClassColors[className] : Color.green;
                            
                            detections.Add(new Detection
                            {
                                x = x,
                                y = y,
                                width = width,
                                height = height,
                                confidence = finalConfidence,
                                classIndex = maxClassIndex,
                                className = className,
                                color = color
                            });
                        }
                    }
                }
            }
        }
        
        return detections;
    }
    
    private List<Detection> ApplyNMS(List<Detection> detections)
{
    // 신뢰도 순으로 정렬
    detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));
    
    // NMS 후 최대 3개 객체만 유지
    List<Detection> result = new List<Detection>();
    bool[] isSuppress = new bool[detections.Count];
    int maxDetections = 3; // 최대 3개 객체로 제한
    
    for (int i = 0; i < detections.Count && result.Count < maxDetections; i++)
    {
        if (isSuppress[i])
            continue;
        
        result.Add(detections[i]);
        
        for (int j = i + 1; j < detections.Count; j++)
        {
            // 같은 클래스이고 IoU가 임계값보다 크면 제거
            if (detections[i].classIndex == detections[j].classIndex)
            {
                float iou = CalculateIoU(detections[i], detections[j]);
                if (iou > iouThreshold)
                {
                    isSuppress[j] = true;
                }
            }
        }
    }

    // NMS 적용 전후 디버그 로그
    Debug.Log($"Before NMS: {detections.Count}, After NMS: {result.Count}");
    return result;
}

    
    private float CalculateIoU(Detection a, Detection b)
    {
        float aLeft = a.x - a.width / 2;
        float aRight = a.x + a.width / 2;
        float aTop = a.y + a.height / 2;
        float aBottom = a.y - a.height / 2;
        
        float bLeft = b.x - b.width / 2;
        float bRight = b.x + b.width / 2;
        float bTop = b.y + b.height / 2;
        float bBottom = b.y - b.height / 2;
        
        float interLeft = Mathf.Max(aLeft, bLeft);
        float interRight = Mathf.Min(aRight, bRight);
        float interTop = Mathf.Min(aTop, bTop);
        float interBottom = Mathf.Max(aBottom, bBottom);
        
        if (interLeft > interRight || interBottom > interTop)
            return 0;
        
        float interArea = (interRight - interLeft) * (interTop - interBottom);
        float aArea = (aRight - aLeft) * (aTop - aBottom);
        float bArea = (bRight - bLeft) * (bTop - bBottom);
        
        return interArea / (aArea + bArea - interArea);
    }
    
    private void CreateBoundingBox(Detection detection)
{
    Camera arCamera = arCameraManager.GetComponent<Camera>();
    
    GameObject bboxObject = new GameObject($"BoundingBox_{detection.className}");
    LineRenderer lineRenderer = bboxObject.AddComponent<LineRenderer>();
    
    lineRenderer.startWidth = 0.02f;
    lineRenderer.endWidth = 0.02f;
    lineRenderer.positionCount = 5;
    lineRenderer.loop = true;
    lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
    lineRenderer.startColor = detection.color;
    lineRenderer.endColor = detection.color;
    lineRenderer.useWorldSpace = true;

    // 3. Bounding Box 좌표 변환 검증을 위한 값 확인
    float halfWidth = detection.width / 2.0f;
    float halfHeight = detection.height / 2.0f;

    // 뷰포트 좌표(0~1 범위)를 계산
    Vector2[] viewportCorners = new Vector2[4];
    viewportCorners[0] = new Vector2(detection.x - halfWidth, detection.y - halfHeight); // 좌하단
    viewportCorners[1] = new Vector2(detection.x + halfWidth, detection.y - halfHeight); // 우하단
    viewportCorners[2] = new Vector2(detection.x + halfWidth, detection.y + halfHeight); // 우상단
    viewportCorners[3] = new Vector2(detection.x - halfWidth, detection.y + halfHeight); // 좌상단

    Vector3[] worldCorners = new Vector3[4];
    for (int i = 0; i < 4; i++)
    {
        // 뷰포트 좌표를 월드 좌표로 변환 (z값은 AR 카메라에서의 거리)
        Vector3 viewportPos = new Vector3(viewportCorners[i].x, viewportCorners[i].y, maxBoundingBoxDistance);
        worldCorners[i] = arCamera.ViewportToWorldPoint(viewportPos);
        Debug.Log($"ViewportPos[{i}]: {viewportPos} -> WorldPos: {worldCorners[i]}");
    }

    lineRenderer.SetPosition(0, worldCorners[0]);
    lineRenderer.SetPosition(1, worldCorners[1]);
    lineRenderer.SetPosition(2, worldCorners[2]);
    lineRenderer.SetPosition(3, worldCorners[3]);
    lineRenderer.SetPosition(4, worldCorners[0]);

    boundingBoxes.Add(bboxObject);

    if (debugTextPrefab != null && uiCanvas != null)
{
    Text classText = Instantiate(debugTextPrefab, uiCanvas.transform);
    classText.text = $"{detection.className} ({detection.confidence:P1})";
    classText.color = Color.red;  
    classText.fontSize = 40;
    Debug.Log($"클래스 이름: {detection.className}, 신뢰도: {detection.confidence:P1}");

    // 변환된 화면 좌표 확인
    Vector3 screenPos = arCamera.ViewportToScreenPoint(new Vector3(detection.x, detection.y + detection.height / 2.0f + 0.05f, maxBoundingBoxDistance));
    Debug.Log($"텍스트 화면 좌표: {screenPos}");

    // 캔버스 Rect 정보 출력
    RectTransform canvasRect = uiCanvas.GetComponent<RectTransform>();
    Debug.Log("캔버스 Rect: " + canvasRect.rect);

    RectTransform textRT = classText.GetComponent<RectTransform>();
    Vector2 anchoredPos;
    if (RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRect, screenPos, arCamera, out anchoredPos))
    {
        textRT.anchoredPosition = anchoredPos;
        Debug.Log("anchoredPosition 설정됨: " + anchoredPos);
    }
    else
    {
        Debug.LogWarning("Screen to LocalPoint 변환 실패!");
    }

    // 임시 테스트: 텍스트를 캔버스 중앙에 배치하여 보이는지 확인
    //textRT.anchoredPosition = Vector2.zero;
    //Debug.Log("임시로 텍스트를 캔버스 중앙에 배치");

    classTexts.Add(classText);
}

}

}
