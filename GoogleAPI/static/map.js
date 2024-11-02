let map;
let data = [];
let markers = [];
let currentSchedule = [];
let currentDaySchedule = [];
let schedules = {};
let geometry;
let routePaths = [];
let directionsService, directionsRenderer;

const colors = { 'mon': '#EA4335', 'tues': '#FF7F0E', 'wed': '#FBBC05', 'thurs': '#34A853',
                 'fri': '#4285F4', 'sat': '#FF69B4', 'sun': '#9B51E0'};

// 初始化地圖
async function initMap() {
    const { Map } = await google.maps.importLibrary("maps");
    const { AdvancedMarkerElement, PinElement } = await google.maps.importLibrary("marker");
    geometry = await google.maps.importLibrary("geometry");
    // 設置地圖中心點和縮放級別
    map = new Map(document.getElementById('map'), {
        center: {lat: 25.1001805, lng: 121.5488112},
        zoom: 12,
        mapId: "DEMO_MAP_ID", // Map ID is required for advanced markers.
    });

    // 根據選擇行程的變化重新加載對應的 JSON 文件
    document.getElementById('schedule').addEventListener('change', function () {
        const selectedSchedule = this.value;
        loadSchedule(selectedSchedule);  // 根據選擇加載對應 JSON 文件
    });

    // 加載三個不同的 JSON 文件
    Promise.all([
        loadScheduleFromJson('schedule1'),
        loadScheduleFromJson('schedule2'),
        loadScheduleFromJson('schedule3')
    ]).then(() => {
        // 所有 JSON 文件加載完成後，生成選擇器
        populateScheduleSelector(Object.keys(schedules));
        // 預設加載第一個行程
        loadSchedule('schedule1');
    });

    // 初始化 Directions API 服務
    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({
        map: map,  // 在地圖上渲染路徑
        suppressMarkers: false,  // 不顯示自動生成的路徑標記
        polylineOptions: {
            strokeColor: "#0000FF",  // 線條顏色
            strokeOpacity: 1.0,      // 線條透明度
            strokeWeight: 10          // 線條寬度
        }
    });
}

initMap();

// 加載 JSON 文件的函數
async function loadScheduleFromJson(scheduleKey) {
    const response = await fetch(`/static/${scheduleKey}.json`);
    const data = await response.json();
    schedules[scheduleKey] = data;
}

// 根據行程鍵生成選擇項
function populateScheduleSelector(scheduleKeys) {
    const scheduleSelect = document.getElementById('schedule');
    // 清空現有的選擇項
    scheduleSelect.innerHTML = ""; 
    scheduleKeys.forEach(key => {
        const option = document.createElement('option');
        option.value = key;
        option.text = key.charAt(0).toUpperCase() + key.slice(1);  // 顯示行程名稱
        scheduleSelect.appendChild(option);
    });

    // 設置事件監聽器：當選擇改變時加載對應行程
    scheduleSelect.addEventListener('change', function () {
        loadSchedule(this.value);  // 根據選擇加載對應行程
    });
}

// 根據選擇的行程加載地圖上的標記
function loadSchedule(scheduleKey) {
    // 清除現有的標記
    markers.forEach(({ marker, label }) => {
        marker.setMap(null);  // 從地圖上移除標記
        label.setMap(null);   // 從地圖上移除標籤
    });
    markers = [];  // 清空標記數組

    // 加載新的行程
    currentSchedule = schedules[scheduleKey];

    // 添加標記到地圖
    let currentDay = null;
    let index = 0;

    currentSchedule.forEach((location) => {
        if (location.currentDay !== currentDay) {
        currentDay = location.currentDay;
        index = 1;
        } else {
        index++;
        }

        const { marker, label } = addMarker(location, index);
        markers.push({ marker, label, day: location.currentDay });
    });
    
    // 初始化頁籤
    document.getElementById("tab1").addEventListener("click", function () {
      switchTab("mon");
    });
    document.getElementById("tab2").addEventListener("click", function () {
      switchTab("tues");
    });

    switchTab("mon");  // 預設顯示 "mon" 行程

}

// Function to add markers to the map
function addMarker(location, number) {
    const { AdvancedMarkerElement, PinElement } = google.maps.marker;
    // 根據 currentDay 設置不同的圖標
    let color = colors[location.currentDay];

    const pin = new PinElement({
        scale: 1.0,
        background: color,
        glyph: number.toString()
    });

    const marker = new AdvancedMarkerElement({
        map: map,
        position: { lat: location.latitude, lng: location.longitude },
        content: pin.element,
        title: location.basicName,
        collisionBehavior: google.maps.CollisionBehavior.OPTIONAL_AND_HIDES_LOWER_PRIORITY
    });

    marker.addListener('click', function () {
        showInfoOnPanel(location);
    });

    // 使用 OverlayView 顯示名稱標籤
    const LabelOverlay = function (position, name, map) {
        this.position = position;
        this.name = name;
        this.map = map;
        this.div = null;
        this.setMap(map); // 在地圖上初始化
    };

    LabelOverlay.prototype = new google.maps.OverlayView();

    LabelOverlay.prototype.onAdd = function () {
        const div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.fontSize = '16px';
        div.style.color = color;
        div.style.padding = '2px 5px';
        div.style.textShadow = `
            -1px -1px 2px white,  
            1px -1px 2px white,
            -1px  1px 2px white,
            1px  1px 2px white
        `; // 添加白色描邊效果
        div.style.maxWidth = '150px';            // 最大寬度限制，避免過長
        div.style.wordBreak = 'break-word';      // 自動換行
        div.style.whiteSpace = 'normal';         // 允許正常換行
        div.style.textAlign = 'center';          // 文字居中對齊
        div.style.fontWeight = 'bold';           // 設置文字為粗體
        div.innerHTML = this.name;

        this.div = div;

        const panes = this.getPanes();
        panes.overlayLayer.appendChild(div);
    };

    LabelOverlay.prototype.draw = function () {
        const overlayProjection = this.getProjection();
        const position = overlayProjection.fromLatLngToDivPixel(this.position);

        const div = this.div;
        if (div) {
            // 設置 div 的位置（將其置於標記的上方）
            div.style.left = position.x + 'px';
            div.style.top = (position.y - 35) + 'px';  // 調整標籤位置
            div.style.transform = 'translate(-50%, -100%)';  // 將文本居中對齊到標記的中間
        }
    };

    LabelOverlay.prototype.onRemove = function () {
        // 從地圖上移除 div
        if (this.div) {
            this.div.parentNode.removeChild(this.div);
            this.div = null;
        }
    };

    // 創建標籤並顯示在標記上方
    const label = new LabelOverlay(new google.maps.LatLng(location.latitude, location.longitude), location.basicName, map);

    return { marker, label };
}

// 將資訊顯示在左側的面板中
function showInfoOnPanel(location) {
    // 自定義 InfoWindow 內容
    const infoContent = `
        <div style="font-family: Arial, sans-serif; max-width: 300px;">
            <div style="display: flex; justify-content: center; align-items: center;">
                <img src="${location.img}" alt="Image" style="width: 100%; height: auto; border-radius: 5px;">
            </div>
            <h2 style="font-size: 20px; margin: 10px 0;">${location.basicName}</h2>
            <div style="font-size: 18px; color: gray; margin-bottom: 5px;">
                ${location.rating} ★ (Reviews: ${location.review})
            </div>
            <div style="font-size: 16px; color: black; margin-bottom: 5px;">
                ${location.label} · $${location.price}
            </div>
            <div style="font-size: 16px; color: gray;">
                ${location.address}
            </div>
        </div>
    `;  
    document.getElementById('popupContent').innerHTML = infoContent;
    document.getElementById('popupWindow').style.display = 'block'; // 顯示彈窗
    document.getElementById('routeSetting').innerHTML = '';
}

// 關閉彈窗
function closePopup() {
    directionsRenderer.setMap(null);
    document.getElementById('popupWindow').style.display = 'none';
}

// 切換頁籤
async function switchTab(day) {
    const tabs = document.getElementsByClassName("tab");
    for (let i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove("active");
    }

    if (day === "mon") {
        document.getElementById("tab1").classList.add("active");
    } else if (day === "tues") {
        document.getElementById("tab2").classList.add("active");
    }

    updateCurrentDaySchedule(day);
    updateMarkerOpacity(day);
    clearAllRoutes();
    await computeRouteForSchedule();
    showSchedule();
}

function updateCurrentDaySchedule(day) {
    currentDaySchedule = [];
    currentSchedule.forEach((location) => {
        if (location.currentDay === day) {
            // console.log("Adding location:", location);
            currentDaySchedule.push(location);
        }
    });
}

// 顯示對應天數的行程
function showSchedule() {
    // console.log("Showing schedule for day:", day);
    let content = "";
    currentDaySchedule.forEach((location, index) => {
        // console.log("Adding location:", location.basicName);
        // 顯示每個地點的名稱和時間
        const locationDiv = `
        <div class="location">
            <h3>${location.basicName}</h3>
            <p>Time: ${location.currentTime}</p>
        </div>
        `;

        // 檢查是否存在路徑信息，將其放在單獨的 div 中，並讓 routeInfo 可點擊
        let routeInfo = '';
        if (location.route) {
            routeInfo = `
            <div class="route-info" id="route-info-${index}" style="margin: 5px 0; color: grey; cursor: pointer;">
                <p>Distance: ${location.route.distance}</p>
                <p>Duration: ${location.route.duration}</p>
            </div>
            `;
        }

        // 組合地點和路徑的 div
        content += `
        <div class="schedule-item">
            ${locationDiv}
            ${routeInfo}
        </div>
        <hr>
        `;
    });
    // console.log("Generated content:", content); 
    document.getElementById("scheduleContent").innerHTML = content;

    // 為每個 route-info 添加 click 事件處理器
    currentDaySchedule.forEach((location, index) => {
        if (location.route) {
            document.getElementById(`route-info-${index}`).addEventListener('click', function() {
                addRouteSettings(index);
                showRouteDetails(location.route.result);
            });
        }
    });
}

function addRouteSettings(index) {
    const settingContent = `
        <div>
            <label for="timeType">Choose Time Type:</label>
            <select id="timeType">
                <option value="leaveNow">Leave now</option>
                <option value="departAt">Depart at</option>
                <option value="arriveBy">Arrive by</option>
            </select>

            <label for="timePicker">Time:</label>
            <input type="time" id="timePicker">

            <label for="datePicker">Date:</label>
            <input type="date" id="datePicker">

            <button id="setTimeButton">Set Time</button>
        </div>
    `;
    document.getElementById('routeSetting').innerHTML = settingContent;

    // document.getElementById('routeSettings').addEventListener('change', function () {
    //     //
    // });

    let transitOptions = {};
    document.getElementById('setTimeButton').addEventListener('click', async function() {
        const selectedOption = document.getElementById('timeType').value;
        // 取得用戶選擇的時間與日期
        const time = document.getElementById('timePicker').value; // 格式 HH:mm
        const date = document.getElementById('datePicker').value; // 格式 YYYY-MM-DD
        
        let selectedDate = null;
    
        if (selectedOption === 'leaveNow') {
            // 如果選擇的是 "Leave Now"，使用當前時間
            selectedDate = new Date();
        } else if (time && date) {
            // 將時間與日期合併成一個字符串，並創建 Date 物件
            const dateTimeString = `${date}T${time}:00`; // "YYYY-MM-DDTHH:mm:ss"
            const selectedDate = new Date(dateTimeString);
            if (selectedOption === 'departAt') {
                transitOptions.departureTime = selectedDate;
            } else if (selectedOption === 'arriveBy') {
                transitOptions.arrivalTime = selectedDate;
            }
            
            const origin = currentDaySchedule[index];         // 起點
            const destination = currentDaySchedule[index + 1]; // 終點
            await computeRouteUsingDirectionsAPI(origin, destination, transitOptions);
            routePaths[index].setMap(null);
            routePaths[index] = origin.route.routePath;
            routePaths[index].setMap(map);
            showRouteDetails(origin.route.result);

            console.log('Selected date and time:', selectedDate);
        } else {
            console.log('Please select date and time.');
        }
    });
}

// 顯示具體路線的詳細信息
function showRouteDetails(routeResult) {
    console.log("Route:", routeResult); 
    let content = '';

    // 假設從 API 回傳結果中提取 routes[0].legs[0] 作為範例
    const legs = routeResult.routes[0].legs;

    legs.forEach((leg, legIndex) => {
        content += `<div class="leg-info">
            <h4>Leg ${legIndex + 1}: ${leg.start_address} to ${leg.end_address}</h4>
            <p>${leg.departure_time ? "Departure time:" + leg.departure_time.text : "Time now:" + new Date()}</p>
            <p>Distance: ${leg.distance.text}, Duration: ${leg.duration.text}</p>
            <p>Estimate arrival time: ${leg.arrival_time ? leg.arrival_time.text : "N/A"}</p>
            <hr>
        </div>`;

        // 遍歷步驟 (steps)
        leg.steps.forEach((step, stepIndex) => {
            // 每個步驟可以是步行，公交，火車等，顯示詳細步驟信息
            const mode = step.travel_mode; // e.g., 'WALKING', 'TRANSIT'
            let stepDetails = '';

            if (mode === 'WALKING') {
                stepDetails = `
                <div class="step-info">
                    <p><strong>Walk:</strong> ${step.distance.text}, ${step.duration.text}</p>
                    <p>Reach: ${step.end_location.lat}, ${step.end_location.lng}</p>
                    <p>Instruction: ${step.instructions}</p>
                </div>
                `;
            } else if (mode === 'TRANSIT') {
                const transit = step.transit;
                stepDetails = `
                <div class="step-info">
                    <p><strong>${transit.line.vehicle.name}:</strong> ${transit.line.short_name}</p>
                    <p>From ${transit.departure_stop.name} to ${transit.arrival_stop.name}</p>
                    <p>Distance: ${step.distance.text}, Duration: ${step.duration.text}</p>
                    <p>Departure time: ${transit.departure_time.text}, Arrival time: ${transit.arrival_time.text}</p>
                </div>
                `;
            }

            content += stepDetails + '<hr>';
        });
    });

    document.getElementById('popupContent').innerHTML = content;
    document.getElementById('popupWindow').style.display = 'block';

    directionsRenderer.setMap(map);
    directionsRenderer.setDirections(routeResult);
}

// 更新標記透明度
function updateMarkerOpacity(selectedDay) {
    markers.forEach(({ marker, label, day }) => {
        if (day === selectedDay) {
            // 顯示當前天數的標記
            marker.content.style.opacity = 1;
            // label.div.style.opacity = 1;
        } else {
            // 使其他天數的標記變得半透明
            marker.content.style.opacity = 0.3;
            // label.div.style.opacity = 0.3;
        }
    });
}

async function computeRouteForSchedule() {
    // 從第一個地點到第二個，第二個到第三個，依此類推
    for (let i = 0; i < currentDaySchedule.length - 1; i++) {
        const origin = currentDaySchedule[i];         // 起點
        const destination = currentDaySchedule[i + 1]; // 終點

        // 檢查 currentSchedule 中是否已經有預計算好的路徑
        if (origin.route && origin.route.routePath) {
            console.log(`Using cached route from ${origin.basicName} to ${destination.basicName}`);
            origin.route.routePath.setMap(map);  // 使用預計算好的路徑
        } else {
            console.log(`Computing route from ${origin.basicName} to ${destination.basicName}`);
            // computeRouteUsingRoutesAPI(origin, destination);
            await computeRouteUsingDirectionsAPI(origin, destination);
            routePaths.push(origin.route.routePath);
        }
    }
}

async function computeRouteUsingDirectionsAPI(origin, destination, options=null) {
    return new Promise((resolve, reject) => {
        directionsService.route(
            {
                origin: { lat: origin.latitude, lng: origin.longitude},
                destination: { lat:destination.latitude, lng: destination.longitude},
                travelMode: google.maps.TravelMode.TRANSIT,  // 設置出行模式
                transitOptions: options
            },
            (result, status) => {
                if (status === "OK") {
                    console.log("Route found:", result);
                    // 在地圖上顯示路徑
                    // directionsRenderer.setDirections(result);
                    const route = result.routes[0];
                    const path = route.overview_path;
                    const routePath = createRoute(path, colors[origin.currentDay]);
                    origin.route = { 
                        result: result, routePath: routePath, 
                        distance: route.legs[0].distance.text, 
                        duration: route.legs[0].duration.text
                    };
                    resolve(result);  // 成功返回
                } else {
                    console.error("Directions request failed due to " + status);
                    reject(status);  // 返回錯誤
                }
            }
        );
    });
}

async function computeRouteUsingRoutesAPI(origin, destination) {
    const apiUrl = `https://routes.googleapis.com/directions/v2:computeRoutes?key=${apiKey}`;
    // 路徑請求數據
    const body = {
      origin: {
        location: {
          lat_lng: {
            latitude: origin.latitude,
            longitude: origin.longitude,
          },
        },
      },
      destination: {
        location: {
          lat_lng: {
            latitude: destination.latitude,
            longitude: destination.longitude,
          },
        },
      },
      travelMode: "TRANSIT",
      computeAlternativeRoutes: true,
      routeModifiers: {
          avoidTolls: false,
          avoidHighways: false,
          avoidFerries: false
      },
      languageCode: "en-US",
      units: "IMPERIAL"
    };
  
    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': apiKey,
                'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
            },
            body: JSON.stringify(body)
        });

        const result = await response.json();
        console.log("Route found:", result);
        if (result.routes && result.routes.length > 0) {
            const route = result.routes[0];
            const path = geometry.encoding.decodePath(route.polyline.encodedPolyline);
            // console.log("Route found:", route);
            // 保存計算好的路徑到 currentSchedule
            const routePath = createRoute(path, colors[origin.currentDay]);
            routePaths.push(routePath);
            origin.route = { routePath: routePath, duration: route.duration, distance: route.distanceMeters };
        } else {
            console.error("No route found");
        }
    } catch (error) {
        console.error("Error fetching the route:", error);
    }
}

function createRoute(path, color) {
    const routePath = new google.maps.Polyline({ // 使用 Polyline 在地圖上繪製路徑
        path: path,
        geodesic: true,
        strokeColor: color, // 路徑顏色
        strokeOpacity: 0.7, // 不透明度
        strokeWeight: 8, // 線條粗細
    });

    routePath.setMap(map); // 將路徑添加到地圖
    return routePath;
}

// 刪除所有路徑
function clearAllRoutes() {
    routePaths.forEach(routePath => {
        routePath.setMap(null);  // 從地圖上移除每條路徑
    });
    routePaths = [];  // 清空數組
}