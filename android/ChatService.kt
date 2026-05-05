package com.example.escombot

import com.google.gson.annotations.SerializedName
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.http.GET

data class HistoryItem(
    @SerializedName("user") val user: String,
    @SerializedName("bot") val bot: String
)

data class ChatRequest(
    @SerializedName("question") val question: String,
    @SerializedName("history") val history: List<HistoryItem> = emptyList()
)

data class ChatResponse(
    @SerializedName("answer") val answer: String,
    @SerializedName("status") val status: String = "ok"
)

data class HealthResponse(
    @SerializedName("status") val status: String,
    @SerializedName("chain_loaded") val chainLoaded: Boolean
)

data class SuggestionResponse(
    @SerializedName("suggestions") val suggestions: List<String>
)

interface ChatService {
    @POST("chat")
    fun getResponse(@Body request: ChatRequest): Call<ChatResponse>

    @GET("suggestions")
    fun getSuggestions(): Call<SuggestionResponse>

    @GET("health")
    fun health(): Call<HealthResponse>
}
