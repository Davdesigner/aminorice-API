"""
Test script for AminoRice API
Run this after starting the API server
"""

import requests
import json
import os

BASE_URL = "https://mission-capstone-kj09.onrender.com"

def test_root():
    print("\n[TEST] Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_register():
    print("\n[TEST] Testing User Registration...")
    user_data = {
        "full_name": "Test User",
        "email": "test@example.com",
        "password": "testpass123",
        "phone": "+1234567890"
    }
    
    response = requests.post(f"{BASE_URL}/register", json=user_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 201

def test_login():
    print("\n[TEST] Testing User Login...")
    login_data = {
        "email": "test@example.com",
        "password": "testpass123"
    }
    
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"[SUCCESS] Login successful!")
        print(f"Token: {data['access_token'][:50]}...")
        return data['access_token']
    else:
        print(f"[FAILED] Login failed: {response.json()}")
        return None

def test_profile(token):
    print("\n[TEST] Testing Get Profile...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/profile", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_update_profile(token):
    print("\n[TEST] Testing Update Profile...")
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "full_name": "Updated Test User",
        "phone": "+9876543210"
    }
    
    response = requests.put(f"{BASE_URL}/profile", headers=headers, params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    print("\n[TEST] Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction(token, image_path):
    print("\n[TEST] Testing Rice Quality Prediction...")
    
    if not os.path.exists(image_path):
        print(f"[SKIPPED] Test image not found at: {image_path}")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    with open(image_path, 'rb') as f:
        files = {'file': ('test_rice.png', f, 'image/png')}
        response = requests.post(f"{BASE_URL}/predict", headers=headers, files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n[PREDICTION RESULTS]")
        
        # Sample Information
        sample_info = data['sample_information']
        print(f"\nSample ID: {sample_info['sample_id']}")
        print(f"Scan ID: {sample_info['scan_id']}")
        print(f"Image URL: {sample_info['image_url'][:60]}...")
        
        # Grain Characteristics
        grains = data['grain_characteristics']
        print(f"\n[GRAIN CHARACTERISTICS]")
        print(f"Total Grains: {grains['total_grains']:.0f}")
        print(f"Broken Grains: {grains['broken_grains']:.0f}")
        print(f"Long Grains: {grains['long_grains']:.0f}")
        print(f"Medium Grains: {grains['medium_grains']:.0f}")
        
        # Defective Grains
        defects = data['defective_grains']
        print(f"\n[DEFECTIVE GRAINS]")
        print(f"Black: {defects['black_grains']:.0f}")
        print(f"Chalky: {defects['chalky_grains']:.0f}")
        print(f"Red: {defects['red_grains']:.0f}")
        print(f"Yellow: {defects['yellow_grains']:.0f}")
        print(f"Green: {defects['green_grains']:.0f}")
        print(f"Total Defective: {defects['total_defective']:.0f}")
        
        # Grain Measurements
        measurements = data['grain_measurements']
        print(f"\n[GRAIN MEASUREMENTS]")
        print(f"Average Length: {measurements['average_length']:.3f} mm")
        print(f"Average Width: {measurements['average_width']:.3f} mm")
        print(f"Length/Width Ratio: {measurements['length_width_ratio']:.3f}")
        
        # Color Characteristics
        color = data['color_characteristics']
        print(f"\n[COLOR CHARACTERISTICS]")
        print(f"Average L (Lightness): {color['average_L']:.2f}")
        print(f"Average a (Green-Red): {color['average_a']:.2f}")
        print(f"Average b (Blue-Yellow): {color['average_b']:.2f}")
        
        # Conclusion
        conclusion = data['conclusion']
        print(f"\n[CONCLUSION]")
        print(f"Broken Grain Percentage: {conclusion['broken_grain_percentage']:.2f}%")
        print(f"Defective Grain Percentage: {conclusion['defective_grain_percentage']:.2f}%")
        print(f"Overall Quality Category: {conclusion['overall_quality_category']}")
        print(f"Description: {conclusion['quality_description']}")
        
        # Store scan_id for later tests
        scan_id = sample_info['scan_id']
        
        return True, scan_id
    else:
        print(f"[FAILED] Prediction failed: {response.json()}")
        return False, None

def test_scan_history(token):
    print("\n[TEST] Testing Get Scan History...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/scans", headers=headers, params={"limit": 10})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        scans = response.json()
        print(f"[SUCCESS] Found {len(scans)} scans in history")
        
        if scans:
            print("\n[RECENT SCANS]")
            for i, scan in enumerate(scans[:3]):  # Show first 3
                print(f"\n  Scan #{i+1}:")
                print(f"    Quality Grade: {scan['quality_grade']}")
                print(f"    Total Count: {scan['total_count']:.0f}")
                print(f"    Broken: {scan['broken_percentage']:.2f}%")
                print(f"    Scanned: {scan['scanned_at']}")
        
        return True, scans[0]['id'] if scans else None
    else:
        print(f"[FAILED] Failed to get scan history: {response.json()}")
        return False, None

def test_scan_details(token, scan_id):
    if not scan_id:
        print("\n[SKIPPED] No scan ID available for detail test")
        return False
    
    print(f"\n[TEST] Testing Get Scan Details (ID: {scan_id[:10]}...)...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/scans/{scan_id}", headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"[SUCCESS] Retrieved scan details")
        grains = data['grain_characteristics']
        conclusion = data['conclusion']
        print(f"  Total Count: {grains['total_grains']:.0f}")
        print(f"  Quality Category: {conclusion['overall_quality_category']}")
        print(f"  Image URL: {data['sample_information']['image_url'][:60]}...")
        return True
    else:
        print(f"[FAILED] Failed to get scan details: {response.json()}")
        return False

def test_chatbot(token):
    print("\n[TEST] Testing Rice Expert Chatbot...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test with a few sample questions
    test_questions = [
        "What is the ideal moisture content for storing rice?",
        "What makes rice chalky?",
        "How is broken rice percentage calculated?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n  Question {i}: {question}")
        chat_data = {"question": question}
        
        response = requests.post(f"{BASE_URL}/chat", headers=headers, json=chat_data)
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            word_count = len(answer.split())
            print(f"  Answer ({word_count} words): {answer}")
            print(f"  Timestamp: {data['timestamp']}")
            
            if i < len(test_questions):  # Don't print separator after last question
                print()
        else:
            print(f"  [FAILED] Chatbot request failed: {response.json()}")
            return False
    
    print(f"\n[SUCCESS] Chatbot test completed")
    return True

def main():
    print("=" * 60)
    print("AminoRice API Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Root endpoint
        if not test_root():
            print("[FAILED] Root endpoint test failed")
            return
        
        # Test 2: Health check
        if not test_health():
            print("[FAILED] Health check failed")
            return
        
        # Test 3: Register user (may fail if user already exists)
        test_register()
        
        # Test 4: Login
        token = test_login()
        if not token:
            print("[FAILED] Login test failed")
            return
        
        # Test 5: Get profile
        if not test_profile(token):
            print("[FAILED] Profile test failed")
            return
        
        # Test 6: Update profile
        if not test_update_profile(token):
            print("[FAILED] Update profile test failed")
            return
        
        # Test 7: Get updated profile
        if not test_profile(token):
            print("[FAILED] Get updated profile test failed")
            return
        
        # Test 8: Prediction (if test images available)
        # Try to find a test image in the data/images folder
        test_image_paths = [
            "../../data/images/ID_000002.png",
            "../../../data/images/ID_000002.png",
            "../../Mission_capstone/data/images/ID_000002.png"
        ]
        
        scan_id = None
        test_image_found = False
        for image_path in test_image_paths:
            if os.path.exists(image_path):
                success, scan_id = test_prediction(token, image_path)
                test_image_found = True
                break
        
        if not test_image_found:
            print("\n[INFO] No test images found. Skipping prediction and scan history tests.")
            print("[INFO] To test prediction, run with an image path:")
            print("       python test_api.py <path_to_rice_image>")
        else:
            # Test 9: Get scan history
            success, history_scan_id = test_scan_history(token)
            if success and not scan_id:
                scan_id = history_scan_id
            
            # Test 10: Get scan details
            if scan_id:
                test_scan_details(token, scan_id)
        
        # Test 11: Rice Expert Chatbot
        test_chatbot(token)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to API")
        print("Make sure the API server is running:")
        print("   uvicorn app:app --reload")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")

if __name__ == "__main__":
    main()
