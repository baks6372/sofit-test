use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::Read;
use futures::{sink::SinkExt, stream::StreamExt};
use hyper::{Body, Request, Response, Method, StatusCode, header};
use hyper::upgrade::Upgraded;
use hyper_tungstenite::{tungstenite, HyperWebsocket, WebSocketStream};
use tungstenite::Message;
use tokio::time::Duration;
use serde::Deserialize;

use opencv::videoio::{VideoCapture, CAP_ANY,  CAP_PROP_FPS};
use opencv::prelude::*;
use opencv::imgcodecs::IMWRITE_JPEG_QUALITY;
use opencv::imgproc::INTER_LINEAR;
use opencv::core::Mat;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

type VideoFrame = Arc<Mutex<Mat>>;
type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const WEB_SERVER_FILES_PATH: &'static str = "";

#[derive(Deserialize, Debug)]
struct VideoStreamingConfig {
    width: u32,
    height: u32,
    fps: u8,
    quality: u8,
}

impl Default for VideoStreamingConfig {
    fn default() -> Self {
        VideoStreamingConfig {
            width: 800,
            height: 600,
            fps: 15,
            quality: 80,
        }
    }
}

fn get_index_page(webserver_files_path: &str) -> Result<Response<Body>, Error> {
    let index_file_path = webserver_files_path.to_owned() + "index.html";
    let mut index_file = File::open(&index_file_path).unwrap();

    let mut index = String::new();
    index_file.read_to_string(&mut index).unwrap();

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/html")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(Body::from(index))
        .unwrap();
    Ok(response)
}

fn create_response_not_found() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(Body::empty())
        .unwrap()
}

async fn handle_request(mut request: Request<Body>, db: VideoFrame) -> Result<Response<Body>, Error> {
    if hyper_tungstenite::is_upgrade_request(&request) {
        let (response, websocket) = hyper_tungstenite::upgrade(&mut request, None)?;
        tokio::spawn(async move {
            if let Err(e) = serve_websocket(websocket, db).await {
                eprintln!("Error in websocket connection: {}", e);
            }
        });
        Ok(response)
    } else {
        match (request.method(), request.uri().path()) {
            (&Method::GET, "/") => {
                get_index_page(WEB_SERVER_FILES_PATH)
            }
            _ => { Ok(create_response_not_found()) },
        }
    }
}
async fn interval_tick_handler(
    websocket_stream: &mut WebSocketStream<Upgraded>,
    video_frame: &VideoFrame,
    video_stream_config: &VideoStreamingConfig
) -> Result<(), Error> {
    let mut frame = Mat::default();
    {
        let video_frame = video_frame.lock().unwrap();
        frame.clone_from(&video_frame);
    }
    let mut resized_frame = Mat::default();
    let (orig_height, orig_width) = (frame.size().unwrap().height, frame.size().unwrap().width);
    let heigth_multiply = orig_height as f64 / video_stream_config.height as f64;
    let width_multiply = orig_width as f64 / video_stream_config.width  as f64;
    opencv::imgproc::resize(
        &frame,
        &mut resized_frame,
        opencv::core::Size::default(),
        heigth_multiply,
        width_multiply,
        INTER_LINEAR)
        .expect("Couldn't resize image");
    let mut params = opencv::core::Vector::new();
    params.push(IMWRITE_JPEG_QUALITY);
    params.push(video_stream_config.quality as i32);
    let mut output_jpeg_data = opencv::core::Vector::new();
    opencv::imgcodecs::imencode(
        ".jpg",
        &resized_frame,
        &mut output_jpeg_data,
        &params
    ).expect("Couldn't change image quality");
    websocket_stream.send(Message::binary(output_jpeg_data.as_slice())).await?;
    Ok(())
}

async fn serve_websocket(websocket: HyperWebsocket, video_frame: VideoFrame) -> Result<(), Error> {
    let mut websocket_stream = websocket.await?;
    let mut video_stream_config = VideoStreamingConfig::default();
    let interval_ms = 1000 / video_stream_config.fps as u64;
    let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                interval_tick_handler(&mut websocket_stream, &video_frame, &video_stream_config).await?;
            }
            ws_msg = websocket_stream.next() => {
                match ws_msg {
                    Some(Ok(msg)) => match msg {
                        Message::Binary(data) => println!("binary {:?}", data),
                        Message::Text(data) => {
                            if let Ok(input_config) = serde_json::from_str::<VideoStreamingConfig>(data.as_str()) {
                                println!("Received input configuration: {:?}", input_config);
                                if input_config.fps.ne(&video_stream_config.fps) {
                                    let interval_ms = 1000 / input_config.fps as u64;
                                    interval = tokio::time::interval(Duration::from_millis(interval_ms));
                                    interval.reset();
                                }
                                video_stream_config = input_config;
                            }
                        },
                        Message::Ping(data) => println!("Ping {:?}", data),
                        Message::Pong(data) => println!("Pong {:?}", data),
                        Message::Close(data) => { println!("Close {:?}", data); break; },
                        Message::Frame(_) => unreachable!(),
                    },
                    Some(Err(_)) => { println!("server went away"); break; },
                    _ => {},
                }
            }
        }
    }
    Ok(())
}

async fn video_streamer(input_file_path: String, shared_video_frame: VideoFrame) -> Result<(), Error> {
    loop {
        let mut reader = VideoCapture::from_file(
            input_file_path.as_str(),
            CAP_ANY
        ).expect("Couldn't create VideoCapture for input file");
        let fps = reader.get(CAP_PROP_FPS).expect("Unable to get fps from video file");
        let interval_ms = 1000 / fps as u64;
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let mut frame = Mat::default();
                    let mut video_frame = shared_video_frame.lock().unwrap();
                    match reader.read(&mut frame) {
                            Ok(false) => break,
                            _ => video_frame.clone_from(&frame),
                    }
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let (
        input_file_path,
        web_server_port
    ) = parse_args();

    let http_port = web_server_port;
    let addr = ([0, 0, 0, 0], http_port).into();
    let video_frame = Arc::new(Mutex::new(Mat::default()));

    let shared_video_frame = video_frame.clone();
    tokio::spawn(async move {
        if let Err(e) = video_streamer(input_file_path, shared_video_frame).await {
            eprintln!("Error in video streamer: {}", e);
        }
    });

    let service = hyper::service::make_service_fn(move |_connection| {
        let inner_video_frame = video_frame.clone();
        async move {
            Ok::<_, hyper::Error>(hyper::service::service_fn(move |req| {
                handle_request(
                    req, inner_video_frame.clone()
                )
            }))
        }
    });
    let server = hyper::Server::bind(&addr).serve(service);
    server.await?;
    Ok(())
}

fn parse_args() -> (
    String,
    u16
) {
    let matches = App::new("RealTrac CAS web server")
        .arg(
            Arg::with_name("input-file-path")
                .short("i")
                .long("input-file")
                .value_name("INPUT FILE PATH")
                .help("Path to video file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("webserver-port")
                .short("p")
                .long("webserver-port")
                .value_name("WEB SERVER PORT")
                .help("Web server port")
                .takes_value(true),
        )
        .get_matches();

    let input_file_path = matches.value_of("input-file-path").expect("enter input video file path").to_owned();
    let web_server_port = value_t!(matches, "webserver-port", u16).unwrap_or(80);

    (
        input_file_path,
        web_server_port
    )
}
