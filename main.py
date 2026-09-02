import 'dart:convert';
import 'package:http/http.dart' as http;

class PredictionResult {
  final String risk;
  final double probability;
  final double baseProbability;
  final double timeMultiplier;
  final String location;
  final String driver;
  final String timeProfile;
  final int hourUsed;
  final int minuteUsed;
  final bool usedMlModel;
  final bool usedFallbackTerrain;
  final double latitude;
  final double longitude;

  PredictionResult({
    required this.risk,
    required this.probability,
    required this.baseProbability,
    required this.timeMultiplier,
    required this.location,
    required this.driver,
    required this.timeProfile,
    required this.hourUsed,
    required this.minuteUsed,
    required this.usedMlModel,
    required this.usedFallbackTerrain,
    required this.latitude,
    required this.longitude,
  });

  factory PredictionResult.fromJson(
    Map<String, dynamic> json,
  ) {
    return PredictionResult(
      risk: json['risk']?.toString() ?? 'UNKNOWN',

      probability:
          (json['probability'] as num?)?.toDouble() ?? 0.0,

      baseProbability:
          (json['base_probability'] as num?)?.toDouble() ?? 0.0,

      timeMultiplier:
          (json['time_multiplier'] as num?)?.toDouble() ?? 1.0,

      location:
          json['location']?.toString() ?? 'Unknown',

      driver:
          json['driver']?.toString() ?? 'Unknown',

      timeProfile:
          json['time_profile']?.toString() ?? 'mixed',

      hourUsed:
          (json['hour_used'] as num?)?.toInt() ?? 0,

      minuteUsed:
          (json['minute_used'] as num?)?.toInt() ?? 0,

      usedMlModel:
          json['used_ml_model'] == true,

      usedFallbackTerrain:
          json['used_fallback_terrain'] == true,

      latitude:
          (json['lat'] as num?)?.toDouble() ?? 0.0,

      longitude:
          (json['lon'] as num?)?.toDouble() ?? 0.0,
    );
  }

  String get timeUsed {
    final hour =
        hourUsed.toString().padLeft(2, '0');

    final minute =
        minuteUsed.toString().padLeft(2, '0');

    return '$hour:$minute';
  }
}


class PredictionService {
  static const String baseUrl =
      'https://hwc-backend-fixed.onrender.com';


  static Future<PredictionResult?> predict(
    double latitude,
    double longitude, {
    int? hour,
    int? minute,
  }) async {

    try {

      final uri = Uri.parse(
        '$baseUrl/predict'
      ).replace(
        queryParameters: {
          'lat': latitude.toString(),
          'lon': longitude.toString(),

          if (hour != null)
            'hour': hour.toString(),

          if (minute != null)
            'minute': minute.toString(),
        },
      );

      final response = await http.get(
        uri,
        headers: {
          'Accept': 'application/json',
        },
      ).timeout(
        const Duration(seconds: 45),
      );

      if (response.statusCode != 200) {
        return null;
      }

      final data =
          jsonDecode(response.body);

      if (data is! Map<String, dynamic>) {
        return null;
      }

      return PredictionResult.fromJson(
        data,
      );

    } catch (e) {

      print(
        'Prediction error: $e'
      );

      return null;
    }
  }
}
