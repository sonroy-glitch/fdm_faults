export type ClassProbability = {
  label: string;
  confidence: number;
};

export type Prediction = {
  prediction: string;
  confidence: number;
  probabilities: ClassProbability[];
  engine: string;
  latency_ms: number;
};

export type ApiError = { error: string };
