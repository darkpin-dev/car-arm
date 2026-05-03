#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_flash.h"

static const char *TAG = "N16R8_TEST";

void app_main(void)
{
    ESP_LOGI(TAG, "============================================");
    ESP_LOGI(TAG, "ESP32-S3 N16R8 System Test Started");
    ESP_LOGI(TAG, "============================================");

    // 1. PSRAM(외부 메모리) 용량 확인
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    
    if (psram_size > 0) {
        ESP_LOGI(TAG, "PSRAM 감지됨: 총 %u bytes (%.2f MB)", psram_size, (float)psram_size / (1024 * 1024));
        ESP_LOGI(TAG, "현재 사용 가능한 PSRAM: %u bytes", free_psram);
    } else {
        ESP_LOGW(TAG, "PSRAM이 감지되지 않았습니다! 설정을 확인하세요.");
    }

    // 2. 내부 RAM 확인
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "사용 가능한 내부 RAM: %u bytes", internal_free);

    // 3. Flash 크기 확인
    uint32_t flash_size;
    if (esp_flash_get_size(NULL, &flash_size) == ESP_OK) {
        ESP_LOGI(TAG, "Flash 크기: %u bytes (%.2f MB)", flash_size, (float)flash_size / (1024 * 1024));
    }

    // 4. PSRAM 할당 테스트 (1MB 할당해보기)
    if (psram_size > 0) {
        void* test_ptr = heap_caps_malloc(1024 * 1024, MALLOC_CAP_SPIRAM);
        if (test_ptr) {
            ESP_LOGI(TAG, "PSRAM에 1MB 할당 성공!");
            free(test_ptr);
        } else {
            ESP_LOGE(TAG, "PSRAM 1MB 할당 실패!");
        }
    }

    ESP_LOGI(TAG, "============================================");
    ESP_LOGI(TAG, "루프 시작 - 1초마다 하트비트 출력");
    
    int count = 0;
    while (1) {
        ESP_LOGI(TAG, "상태 보고 [%d]: 실행 중...", count++);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
