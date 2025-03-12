## Unity Version & Installed Plugins
- **Unity Version**: Unity 6.6000.0.41f1
- **Installed Plugins**:
  - XR Interaction Toolkit (version 3.0.7)
  - AR Foundation (version 6.0.5)
  - Oculus XR Plugin (version 4.5.0)
 
<br>

## Issues and Solutions
<details>
<summary>1. 카메라 활성화 표시되지만 화면에 아무것도 보이지 않음</summary>

- iOS 빌드를 통해 카메라 연결을 확인했는데, 카메라는 정상적으로 활성화되었다는 표시가 뜨는 반면, 화면에는 노란색 화면만 표시됨.

- **해결 방법**:  
  **AR Background Renderer Feature**가 제대로 설정되지 않아서 발생한 문제. 아래 설정을 확인하면 해결됨:
  1. **Settings > Mobile_Renderer**로 이동
  2. **Renderer Feature**에서 **AR Background Renderer Feature**가 추가되어 있는지 확인
  
  🪄 이 해결 방법은 다음 링크에서 참고함: [Unity Discussions Link](https://discussions.unity.com/t/yellow-screen-when-doing-ar/1542033/4)

</details>
