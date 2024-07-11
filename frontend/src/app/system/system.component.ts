import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import {Router} from "@angular/router";
import { AppService} from "../app.service";

@Component({
  selector: 'app-system',
  templateUrl: './system.component.html',
  styleUrls: ['./system.component.scss']
})
export class SystemComponent implements OnInit {
  public data: any;
  //@ts-ignore
  user = JSON.parse(localStorage.getItem('user'));
  tag = "";
  code = "";
  code1 ="";
  code2 = "";
  modalActive = false;
  isAdded = false;
  loader = true;
  invoices:any=[];
  bert = '';
  sgd2='';
  sgd4='';
  sgdpipline='';
  maincode = '';
  code6='';
  constructor(private router: Router, private changeDetection: ChangeDetectorRef,private service: AppService) { }

  ngOnInit(): void {
    if (!this.user){
      this.router.navigate(['auth']);
    }
    this.service.getlist().subscribe(value => {
      this.data=value;
      this.invoices = this.data.items
    })
  }
  openModal(){
    this.loader=true;
    this.modalActive = true;
    this.service.getCode(this.tag).subscribe(value => {
      this.data=value;
      console.log(this.data);
      this.bert=this.data.items.bert;
      this.sgd2 = this.data.items.sgd;
      this.sgd4 = this.data.items.sgd4;
      this.sgdpipline = this.data.items.sgdpipline;
      this.maincode=this.data.items.main;
      this.code6 = this.data.items.code;
      this.loader=false;
    })
  }
  closeModal(){
    this.modalActive = false;

  }
  add(){
    let d = new Date();
    let dat = d.toLocaleString('ru', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      })
    this.invoices.unshift({
      data: dat,
      autor: this.user.name + " " + this.user.surname,
      name: this.tag,
      code: this.maincode+' '+this.code6
    })
    this.service.setlist(this.tag,this.user.name + " " + this.user.surname,this.maincode+' '+this.code6,dat).subscribe(value => {
      console.log(value)
    });
    this.modalActive = false;
    this.isAdded = true;
    setInterval(() => {this.isAdded = false}, 2000);

  }

  search(event: any) {
    this.service.onCange(this.tag).subscribe(value => {
      this.data = value;
      this.code = this.data.items.code;
      this.code1=this.data.items.main.code1;
      this.code2=this.data.items.main.code2;
    })
  }

  logout(){
    localStorage.removeItem('user');
    this.router.navigate(['auth']);
  }

}
