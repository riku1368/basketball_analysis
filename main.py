import os
import json
import argparse
from datetime import datetime
import pandas as pd
from utils import read_video, save_video, get_video_fps
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save JSON/CSV analysis results')
    parser.add_argument('--save_data', action='store_true',
                        help='Save analysis data as JSON and CSV files')
    parser.add_argument('--skip_video', action='store_true',
                        help='Skip video output (only save data)')
    return parser.parse_args()

def convert_to_json_serializable(obj):
    """numpy型やその他の型をJSON互換の型に変換"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_analysis_results(output_dir, video_frames, **data):
    """分析結果をJSONとCSV形式で保存する"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # データをJSON互換形式に変換
    json_compatible_data = convert_to_json_serializable(data)
    
    # 1. JSON保存（Unity用・完全なデータ）
    json_path = os.path.join(output_dir, f'analysis_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_compatible_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ JSON saved: {json_path}")
    
    # 2. CSV保存（データ分析用）
    
    # プレイヤー統計CSV
    if 'player_stats' in data and data['player_stats']:
        csv_path = os.path.join(output_dir, f'player_stats_{timestamp}.csv')
        df = pd.DataFrame(data['player_stats'])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ Player stats CSV saved: {csv_path}")
    
    # パスのCSV
    if 'passes' in data and data['passes']:
        passes_data = data['passes']
        passes_list = []
        
        if isinstance(passes_data, list):
            for frame_num, pass_info in enumerate(passes_data):
                if pass_info != -1:
                    passes_list.append({
                        'frame': frame_num,
                        'player_id': pass_info
                    })
        
        if passes_list:
            passes_csv = os.path.join(output_dir, f'passes_{timestamp}.csv')
            df_passes = pd.DataFrame(passes_list)
            df_passes.to_csv(passes_csv, index=False, encoding='utf-8')
            print(f"✓ Passes CSV saved: {passes_csv} ({len(passes_list)} passes)")
        else:
            print(f"⚠ No passes detected")
    
    # インターセプトのCSV
    if 'interceptions' in data and data['interceptions']:
        interceptions_data = data['interceptions']
        interceptions_list = []
        
        if isinstance(interceptions_data, list):
            for frame_num, interception_info in enumerate(interceptions_data):
                if interception_info != -1:
                    interceptions_list.append({
                        'frame': frame_num,
                        'player_id': interception_info
                    })
        
        if interceptions_list:
            interceptions_csv = os.path.join(output_dir, f'interceptions_{timestamp}.csv')
            df_interceptions = pd.DataFrame(interceptions_list)
            df_interceptions.to_csv(interceptions_csv, index=False, encoding='utf-8')
            print(f"✓ Interceptions CSV saved: {interceptions_csv} ({len(interceptions_list)} interceptions)")
        else:
            print(f"⚠ No interceptions detected")
    
    # 速度・距離のサマリーCSV
    if 'player_stats' in data and data['player_stats']:
        df_stats = pd.DataFrame(data['player_stats'])
        
        summary = df_stats.groupby('player_id').agg({
            'distance': 'sum',
            'speed': 'mean',
            'has_ball': 'sum'
        }).reset_index()
        
        summary.columns = ['player_id', 'total_distance', 'avg_speed', 'ball_possession_frames']
        
        team_mapping = df_stats.groupby('player_id')['team'].first()
        summary['team'] = summary['player_id'].map(team_mapping)
        
        summary_csv = os.path.join(output_dir, f'player_summary_{timestamp}.csv')
        summary.to_csv(summary_csv, index=False, encoding='utf-8')
        print(f"✓ Player summary CSV saved: {summary_csv}")
    
    print(f"\n📁 All data saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Read Video
    video_frames = read_video(args.input_video)

    original_fps = get_video_fps(args.input_video)
    
    ## Initialize Tracker
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )
    
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                )
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
                                                                    )

    # Remove Wrong Ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
   

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )

    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    # Detect Passes
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame,player_tracks)

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # ===== データ保存（オプション）=====
    if args.save_data:
        print("\n" + "=" * 60)
        print("Saving analysis data...")
        print("=" * 60)
        
        # プレイヤー統計をフラット化（CSV用）
        player_stats = []
        try:
            if isinstance(player_tracks, list):
                for frame_num, frame_data in enumerate(player_tracks):
                    if frame_data:
                        for player_id, player_info in frame_data.items():
                            team = 'unknown'
                            if isinstance(player_assignment, list) and frame_num < len(player_assignment):
                                team = player_assignment[frame_num].get(player_id, 'unknown') if isinstance(player_assignment[frame_num], dict) else 'unknown'
                            
                            distance = 0
                            if isinstance(player_distances_per_frame, list) and frame_num < len(player_distances_per_frame):
                                distance = player_distances_per_frame[frame_num].get(player_id, 0) if isinstance(player_distances_per_frame[frame_num], dict) else 0
                            
                            speed = 0
                            if isinstance(player_speed_per_frame, list) and frame_num < len(player_speed_per_frame):
                                speed = player_speed_per_frame[frame_num].get(player_id, 0) if isinstance(player_speed_per_frame[frame_num], dict) else 0
                            
                            has_ball = False
                            if isinstance(ball_aquisition, list) and frame_num < len(ball_aquisition):
                                has_ball = (ball_aquisition[frame_num] == player_id)
                            
                            stat = {
                                'frame': int(frame_num),
                                'player_id': int(player_id),
                                'team': str(team),
                                'distance': float(distance),
                                'speed': float(speed),
                                'has_ball': bool(has_ball)
                            }
                            player_stats.append(stat)
            
            print(f"✓ Generated {len(player_stats)} player stat records")
        except Exception as e:
            print(f"⚠ Warning: Error generating player stats: {e}")
            player_stats = []
        
        # 保存するデータを整理
        analysis_results = {
            'video_info': {
                'input_path': args.input_video,
                'fps': original_fps,
                'total_frames': len(video_frames)
            },
            'player_tracks': player_tracks,
            'ball_tracks': ball_tracks,
            'court_keypoints': court_keypoints_per_frame,
            'player_assignment': player_assignment,
            'ball_possession': ball_aquisition,
            'passes': passes,
            'interceptions': interceptions,
            'tactical_positions': tactical_player_positions,
            'player_distances': player_distances_per_frame,
            'player_speeds': player_speed_per_frame,
            'player_stats': player_stats
        }
        
        save_analysis_results(args.output_dir, video_frames, **analysis_results)

    # ===== Draw output =====
    if not args.skip_video:
        print("\n" + "=" * 60)
        print("Drawing video annotations...")
        print("=" * 60)
        
        # Initialize Drawers
        player_tracks_drawer = PlayerTracksDrawer()
        ball_tracks_drawer = BallTracksDrawer()
        court_keypoint_drawer = CourtKeypointDrawer()
        team_ball_control_drawer = TeamBallControlDrawer()
        frame_number_drawer = FrameNumberDrawer()
        pass_and_interceptions_drawer = PassInterceptionDrawer()
        tactical_view_drawer = TacticalViewDrawer()
        speed_and_distance_drawer = SpeedAndDistanceDrawer()

        ## Draw object Tracks
        output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                        player_tracks,
                                                        player_assignment,
                                                        ball_aquisition)
        output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

        ## Draw KeyPoints
        output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)

        ## Draw Frame Number
        output_video_frames = frame_number_drawer.draw(output_video_frames)

        # Draw Team Ball Control
        output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                            player_assignment,
                                                            ball_aquisition)

        # Draw Passes and Interceptions
        output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                                 passes,
                                                                 interceptions)
        
        # Speed and Distance Drawer
        output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                             player_tracks,
                                                             player_distances_per_frame,
                                                             player_speed_per_frame
                                                             )

        ## Draw Tactical View
        output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                        tactical_view_converter.court_image_path,
                                                        tactical_view_converter.width,
                                                        tactical_view_converter.height,
                                                        tactical_view_converter.key_points,
                                                        tactical_player_positions,
                                                        player_assignment,
                                                        ball_aquisition,
                                                        )

        # Save video
        print("\n" + "=" * 60)
        print("Saving output video...")
        print("=" * 60)
        save_video(output_video_frames, args.output_video, fps=original_fps)
        print(f"✓ Video saved: {args.output_video}")
    else:
        print("\n⚠ Video output skipped (--skip_video flag)")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()